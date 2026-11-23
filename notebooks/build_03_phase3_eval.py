"""Generate ``notebooks/03_phase3_evaluation.ipynb`` (Korean Phase 3 evaluation notebook).

Run from the project root:
    deep_learning/Scripts/python notebooks/build_03_phase3_eval.py
Heavy cells (prediction, timing) are off by default; the notebook reads the cached CSV results.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(r"""# 03. Phase 3 — 평가: 정확도 · 속도 · 강건성

**CSE36501 Deep Learning (UNIST, Fall 2026)** · 정영찬 (20221355), 조수민 (20221371)

Phase 2에서 학습한 모델들을 **같은 규칙으로** 평가해 세 가지 연구 질문에 답합니다.

| 질문 | 이 노트북에서 보는 것 |
|---|---|
| **RQ1** 정확도와 효율은 어떻게 맞바뀌나? | 정확도(mAP) vs 연산량(GFLOPs)·지연시간(ms) 그래프, **파레토 프런트** |
| **RQ2** 어떤 경량화 방법이 정확도를 가장 덜 잃나? | 기준 모델(yolo26n, 640, 전체 프레임) 대비 *정확도 변화* vs *아낀 비용* |
| **RQ3** 다른 도메인·비에도 버티나? | 합성 비(B1~B3), 다른 데이터셋(C), 실제 비(D)에서의 성능 하락률 |

**사용 규칙**
- 위에서부터 순서대로 실행합니다. 무거운 셀(예측·속도 측정)은 `RUN_... = False`가 기본입니다.
- 실제 코드는 `src/eval/`에 있고 노트북은 실행·시각화만 합니다 → 터미널에서도 똑같이 돌릴 수 있습니다.
- 예측 결과는 `outputs/eval/preds/`에 저장(캐시)되어, 새 run이 생겼을 때 **그 run만** 새로 예측합니다.
"""),
    md(r"""## 0. 평가 규칙 (보고서 Experimental Setup에 그대로 씁니다)

1. **테스트셋**: A(RDD 테스트, 맑음), B1/B2/B3(A에 합성 비: 이슬비/폭우/호우), C(MWPD 맑음), D(MWPD 실제 비).
   모두 학습에 쓰지 않은 이미지입니다.
2. **confidence 임계값 고정**: 각 run마다 **RDD 검증셋**에서 F1이 최대가 되는 임계값을 구해, 모든 테스트셋에 **똑같이** 적용합니다.
   테스트셋을 보고 임계값을 고르면 성능이 부풀려지기 때문입니다. (mAP는 임계값과 무관하고, P/R/F1에만 영향)
3. **잘라낸 입력(crop60) 모델도 전체 프레임 정답으로 채점**: 차에서처럼 프레임 아래 60%만 넣고, 예측 박스를 원래 좌표로 되돌려
   **전체 정답**과 비교합니다. 위쪽 40%에 있던 포트홀은 *놓친 것(FN)* 으로 셉니다 → 잘라서 생기는 손해까지 공정하게 반영.
4. **시드 3개 평균 ± 표준편차**. 두 설정의 평균 차이가 **두 표준편차의 합보다 작으면 "n.s."**(유의하지 않음)로 표시합니다.
5. **속도**: 같은 RTX 3090에서 batch 1, 워밍업 50회 후 300장 평균(CPU는 10회 후 100장). 전처리·추론·후처리(NMS)를 모두 포함합니다.
   GFLOPs도 640×640 정사각형이 아니라 **실제로 들어가는 직사각형 입력 크기**로 계산합니다.
6. **크기별 AP**: COCO 기준 small(< 32²px) / medium(< 96²px) / large, 원본 이미지 픽셀 기준 (`pycocotools`).

mAP 수치는 Phase 2 학습 로그의 검증 mAP와 조금 다를 수 있습니다 (평가셋·입력 패딩 방식이 다름). 모든 모델을 **같은 코드로** 평가하므로 비교는 공정합니다.
"""),
    code(r"""import json, os, subprocess, sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml
from matplotlib import font_manager
for name in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = name; break
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

import torch
print("Python  :", sys.version.split()[0], "|", sys.executable)
print("PyTorch :", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음 (CPU)")
ECFG = yaml.safe_load((ROOT / "configs/eval.yaml").read_text(encoding="utf-8"))

def run(module, args=()):
    # src.eval.* 를 별도 프로세스로 실행하고 로그를 보여 줌 (진행 막대는 마지막 상태만)
    cmd = [sys.executable, "-m", module, *args]
    print("$", " ".join(cmd[1:]), flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env)
    for chunk in proc.stdout:
        for part in chunk.split("\r"):
            if part.endswith("\n") and "it/s" not in part and "s/it" not in part:
                print(part, end="", flush=True)
    rc = proc.wait()
    print(f"[종료 코드 {rc}]", "✅" if rc == 0 else "❌ 위 로그의 마지막 오류를 확인하세요")
    return rc
"""),
    md(r"""## 1. 평가 흐름 한눈에 보기"""),
    code(r"""import matplotlib.patches as mp

fig, ax = plt.subplots(figsize=(13, 4.6)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 4.6)
def box(x, y, w, h, text, color):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="0.3"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="0.3"))

box(0.1, 1.9, 1.8, 0.9, "학습된 모델\nbest.pt\n(run × 시드)", "#dbe9f6")
box(2.6, 3.3, 2.4, 0.9, "RDD 검증셋\n→ F1 최대 임계값 고정", "#fde9c9")
box(2.6, 1.9, 2.4, 0.9, "테스트셋 6개\nA · B1 · B2 · B3 · C · D", "#fde9c9")
box(2.6, 0.4, 2.4, 0.9, "A의 이미지 300장\n(batch 1)", "#fde9c9")
box(5.7, 1.9, 2.6, 0.9, "crop 모델이면 아래 60%만 입력\n→ 박스를 원래 좌표로", "#e4f2e1")
box(5.7, 0.4, 2.6, 0.9, "GPU FP32 / FP16, CPU\n전처리+추론+NMS 시간", "#e4f2e1")
box(9.0, 2.6, 3.8, 1.4, "results.csv\nmAP50, mAP50-95, P/R/F1\n크기별 AP (small/medium/large)\n+ A를 FP16으로 한 번 더", "#efe3f5")
box(9.0, 0.4, 3.8, 1.2, "speed.csv\nms/이미지, FPS,\n실제 입력 기준 GFLOPs", "#efe3f5")
arrow(1.9, 2.35, 2.6, 3.75); arrow(1.9, 2.35, 2.6, 2.35); arrow(1.9, 2.35, 2.6, 0.85)
arrow(5.0, 3.75, 9.0, 3.4); arrow(5.0, 2.35, 5.7, 2.35); arrow(8.3, 2.35, 9.0, 3.0); arrow(5.0, 0.85, 5.7, 0.85); arrow(8.3, 0.85, 9.0, 1.0)
ax.text(10.9, 4.35, "→ aggregate.py: 시드 평균±표준편차, n.s., 하락률, 그림", ha="center", fontsize=9, color="0.25")
ax.set_title("Phase 3 평가 파이프라인 (src/eval/evaluate.py · speed.py · aggregate.py)", fontsize=11)
plt.tight_layout(); plt.show()
"""),
    md(r"""## 2. 무엇이 학습되어 있나

`outputs/runs/*/run_meta.json`이 있는 run만 평가합니다. 아직 학습하지 않은 run은 맨 마지막(§12)에 **누락 목록**으로 나옵니다.
"""),
    code(r"""metas = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((ROOT / "outputs/runs").glob("*/run_meta.json"))]
runs = pd.DataFrame(metas)
if runs.empty:
    print("학습된 run이 없습니다. 02 노트북에서 먼저 학습하세요.")
else:
    show = runs[["run_id", "exp", "params_M", "gflops", "epochs_done", "best_epoch", "val_mAP50", "val_mAP50_95", "wall_hours"]]
    display(show.sort_values(["exp", "run_id"]).reset_index(drop=True))
    print(f"학습 완료 run: {len(runs)}개 | 시드별:", runs["seed"].value_counts().sort_index().to_dict())

# E0(RT-DETR)용 학습 설정(AdamW, lr0=1e-4)이 실제로 학습에 전달되는지 확인
from src.train.train import plan_runs
e0 = plan_runs(["E0"], None, None)
ta = getattr(e0[0], "train_args", None) if e0 else None
if ta:
    print("E0 train_args 전달됨:", ta)
else:
    print("⚠️ E0 train_args가 비어 있음: src/train/train.py의 plan_runs()에서 Run(...)에 "
          "train_args=spec.get('train_args', {}) 를 넣어야 lr0=1e-4가 적용됩니다.")
"""),
    md(r"""## 3. 테스트셋 미리 보기

같은 RDD 이미지가 비의 세기(B1→B3)에 따라 어떻게 변하는지, MWPD(C, D)는 어떻게 다른지 봅니다. 초록 상자 = 정답 포트홀.
"""),
    code(r"""import cv2
from src.data.synth_rain import read_list
from src.eval.evaluate import label_path

lists = {k: read_list(ROOT / v) for k, v in ECFG["testsets"].items()}
display(pd.read_csv(ROOT / "outputs/tables/tab_test_sets.csv"))

def n_labels(p):
    return sum(1 for r in label_path(p).read_text().splitlines() if r.strip())

def draw(ax, p, title):
    im = cv2.cvtColor(cv2.imdecode(np.fromfile(p, np.uint8), 1), cv2.COLOR_BGR2RGB)
    H, W = im.shape[:2]; ax.imshow(im)
    for r in label_path(p).read_text().splitlines():
        if r.strip():
            _, cx, cy, w, h = map(float, r.split())
            ax.add_patch(mp.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, fill=False, ec="lime", lw=1.5))
    ax.set_title(title, fontsize=9); ax.axis("off")

i = next(k for k, p in enumerate(lists["A"]) if n_labels(p) >= 2 and "Japan" in p.name)
c = next(p for p in lists["C"] if n_labels(p) >= 1)
d = next(p for p in lists["D"] if n_labels(p) >= 1)
fig, axes = plt.subplots(1, 6, figsize=(18, 3.2))
for ax, k in zip(axes[:4], ["A", "B1", "B2", "B3"]):
    p = next(q for q in lists[k] if q.name == lists["A"][i].name)
    draw(ax, p, f"{k}: {p.parent.parent.name}")
draw(axes[4], c, "C: MWPD 맑음"); draw(axes[5], d, "D: MWPD 비")
plt.tight_layout(); plt.show()
"""),
    md(r"""## 4. 정확도 평가 실행 (`src/eval/evaluate.py`)

- 처음 한 run을 평가하는 데 RTX 3090 기준 **약 1~3분** (검증셋 + 테스트셋 6개 + FP16 한 번, 약 4,900장 예측).
- 한 번 예측한 결과는 캐시되므로 다시 실행하면 **채점만** 해서 몇 초면 끝납니다.
- 새 run(예: 시드 43/44, E0)을 학습한 뒤 이 셀만 다시 실행하면 그 run만 새로 예측합니다.

터미널: `deep_learning\Scripts\python -m src.eval.evaluate`
"""),
    code(r"""RUN_EVAL = False   # True로 바꾸고 실행 (새로 학습된 run이 있을 때)

if RUN_EVAL:
    _ = run("src.eval.evaluate")

res_path = ROOT / ECFG["paths"]["results"]
if res_path.exists():
    res = pd.read_csv(res_path)
    print(f"results.csv: {res['run_id'].nunique()}개 run × {res['testset'].nunique()}개 테스트셋, {len(res)}행")
    display(res[res["testset"] == "A"][["run_id", "precision", "mAP50", "mAP50_95", "P", "R", "F1", "conf_thr"]].round(3))
else:
    res = None
    print("아직 결과가 없습니다 → RUN_EVAL = True 로 실행하세요.")
"""),
    md(r"""## 5. 속도 측정 (`src/eval/speed.py`)

- 시드끼리는 구조가 같으므로 **설정(모델·데이터·입력 크기)마다 한 번**만 잽니다 (KD run은 yolo26n과 같은 구조).
- GPU FP32 / GPU FP16 / CPU FP32 세 가지. 설정 하나에 약 1분, RT-DETR은 CPU에서 더 오래 걸립니다.
- **측정 중에는 다른 GPU 작업(학습, 게임, 브라우저 영상)을 끄세요.** 다른 작업이 돌면 시간이 들쭉날쭉해집니다.

터미널: `deep_learning\Scripts\python -m src.eval.speed`
"""),
    code(r"""RUN_SPEED = False   # True로 바꾸고 실행 (새 설정이 학습되었을 때)

if RUN_SPEED:
    _ = run("src.eval.speed")

spd_path = ROOT / ECFG["paths"]["speed"]
spd = pd.read_csv(spd_path) if spd_path.exists() else pd.DataFrame()
if spd.empty:
    print("속도 결과가 없습니다 → RUN_SPEED = True 로 실행하세요.")
else:
    print("측정 장비:", ", ".join(spd["hardware"].unique()))
    display(spd.pivot_table(index="config", columns=["device", "precision"], values="total_ms").round(2))
"""),
    md(r"""## 6. 집계 (`src/eval/aggregate.py`)

표는 `outputs/tables/tab_accuracy|tab_robustness|tab_efficiency.{csv,tex}`, 그림은 `outputs/figures/fig_pareto|fig_robustness.{pdf,png}`로 저장되어 보고서에 바로 들어갑니다.
"""),
    code(r"""if res is not None:
    _ = run("src.eval.aggregate")
    from src.eval import aggregate as agg
    acc = agg.accuracy_table(res, ECFG["anchor"])
    rob = agg.robustness_table(res)
    eff = agg.efficiency_table(res, spd)
    print("\n▶ 테스트셋 A 정확도 (시드 평균 ± 표준편차) — 기준:", ECFG["anchor"])
    display(acc)
"""),
    md(r"""**표 읽는 법**
- `seeds`가 1이면 아직 시드 42만 학습된 것 → 표준편차가 없어 `vs anchor` 칸이 `n<2`로 나옵니다. 시드 43/44를 학습하면 채워집니다.
- `n.s.` = 기준 모델과의 차이가 시드 간 흔들림보다 작음 → "차이가 있다"고 말하면 안 됨.
- 빈칸 = 차이가 시드 흔들림보다 큼 → 의미 있는 차이.
- `AP_small`은 작은 포트홀(32×32px 미만) 성능 — P2 브랜치(E4)가 노리는 지표입니다.
"""),
    md(r"""## 7. RQ1 — 정확도 vs 효율 (파레토 프런트)

점 하나 = 설정 하나. **왼쪽 위**일수록 좋습니다 (가볍고 정확). 빨간 계단선 = 파레토 프런트: 이 선 위의 모델보다 *더 싸면서 더 정확한* 모델이 없는 것들.
차량에 넣을 후보는 이 선 위에서 고릅니다.

> **GPU 시간이 모델마다 거의 같은 이유 (5~8 ms)**: RTX 3090은 nano·small 모델을 batch 1로 돌리면 계산보다
> *GPU에 명령을 보내는 고정 비용*(커널 실행 오버헤드, 파이썬 처리)이 더 커서, 연산량이 4배 달라도 시간이 비슷합니다.
> 같은 이유로 FP16이 오히려 약간 느립니다(변환 비용 > 절약). 그래서 **효율 비교는 GFLOPs와 CPU 시간**을 주로 봅니다.
> CPU 시간은 연산량에 거의 비례하고, 연산 여유가 적은 차량용 칩의 상황에 더 가깝습니다.
"""),
    code(r"""if res is not None:
    display(eff.round(3))
    fig = agg.fig_pareto(eff); plt.show()
    x = "CPU FP32 ms" if "CPU FP32 ms" in eff else "params_M"
    d = eff.dropna(subset=[x]).sort_values(x)
    front, best = [], -1
    for c, r in d.iterrows():
        if r["mAP50-95 (A)"] > best:
            front.append(c); best = r["mAP50-95 (A)"]
    print(f"파레토 프런트 ({x} 기준, 가벼운 순):", " → ".join(front))
"""),
    md(r"""## 8. RQ2 — 어떤 경량화 방법이 가장 싸게 먹히나

기준(yolo26n · 640 · 전체 프레임)에서 **한 가지만 바꾼** 설정을 비교합니다.
`Δ`는 기준 대비 변화량 (정확도는 음수 = 손해, 비용은 음수 = 절약). **같은 비용을 아낄 때 정확도를 덜 잃는 방법**이 좋은 레버입니다.

| 레버 | 설정 |
|---|---|
| 모델 크기 | yolo26s (더 큼), yolov8n/s (다른 계열) |
| 입력 해상도 | 480, 320 |
| 입력 영역(도로만) | crop60 |
| 경량 모듈 | P2 브랜치 (작은 물체용, 비용 증가) |
| 지식 증류 | yolo26s → yolo26n (추론 비용 동일, 정확도만 변화) |
| 정밀도 | FP16 (같은 가중치, 숫자 표현만 절반) |
"""),
    code(r"""if res is not None:
    anchor = ECFG["anchor"]
    if anchor not in eff.index:
        print("기준 설정이 아직 평가되지 않았습니다:", anchor)
    else:
        a = eff.loc[anchor]
        cost_cols = [c for c in ["GFLOPs_eff", "GPU FP16 ms", "CPU FP32 ms"] if c in eff]
        rows = []
        for c, r in eff.iterrows():
            if c == anchor:
                continue
            row = {"설정": c, "ΔmAP50-95": r["mAP50-95 (A)"] - a["mAP50-95 (A)"]}
            for k in cost_cols:
                row[f"Δ{k} (%)"] = 100 * (r[k] - a[k]) / a[k]
            rows.append(row)
        if "ΔmAP50-95 FP16" in eff:
            row = {"설정": f"{anchor} → FP16", "ΔmAP50-95": a["ΔmAP50-95 FP16"]}
            if "GPU FP16 ms" in eff and "GPU FP32 ms" in eff:
                row["ΔGPU FP16 ms (%)"] = 100 * (a["GPU FP16 ms"] - a["GPU FP32 ms"]) / a["GPU FP32 ms"]
                row["(FP32→FP16 기준)"] = "GPU FP32 대비"
            rows.append(row)
        lever = pd.DataFrame(rows).set_index("설정")
        stats = agg.seed_stats(res, "mAP50_95", "A")
        lever["n.s.?"] = agg.ns_mark(stats, anchor).reindex(lever.index).fillna("")
        display(lever.round(3))

        if "GFLOPs_eff" in eff:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for c, r in eff.iterrows():
                dx = 100 * (r["GFLOPs_eff"] - a["GFLOPs_eff"]) / a["GFLOPs_eff"]
                dy = r["mAP50-95 (A)"] - a["mAP50-95 (A)"]
                ax.scatter(dx, dy, s=40); ax.annotate(c, (dx, dy), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.axhline(0, color="0.5", lw=0.8); ax.axvline(0, color="0.5", lw=0.8)
            ax.set_xlabel("연산량 변화 (%) — 왼쪽일수록 가벼움"); ax.set_ylabel("mAP50-95 변화 — 위일수록 정확")
            ax.set_title("경량화 레버: 왼쪽 위 사분면 = 공짜 점심 (더 가볍고 더 정확)"); ax.grid(alpha=0.3)
            plt.tight_layout(); plt.show()
"""),
    md(r"""## 9. RQ3 — 강건성: 비와 다른 도메인

- `drop_B1~B3` = 합성 비로 인한 하락률 `(A − B)/A` → **순수한 비의 영향** (같은 장면, 비만 추가)
- `drop_C` = 다른 데이터셋으로 갔을 때의 하락률 `(A − C)/A` → **도메인 차이**
- `drop_C→D` = 같은 MWPD 안에서 맑음 → 실제 비 `(C − D)/C` → **실제 비의 영향**

하락률이 **작을수록** 강건합니다. 가벼운 모델이 비에 특히 약한지(정확도 외의 숨은 비용)를 봅니다.
"""),
    code(r"""if res is not None:
    display(rob.round(3))
    fig = agg.fig_robustness(rob); plt.show()
    drops = rob[[c for c in rob if c.startswith("drop")]]
    if not drops.empty:
        fig, ax = plt.subplots(figsize=(1.3 * drops.shape[1] + 3, 0.42 * len(drops) + 1.2))
        im = ax.imshow(drops.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(drops.shape[1]), drops.columns, fontsize=8); ax.set_yticks(range(len(drops)), drops.index, fontsize=8)
        for (r, c), v in np.ndenumerate(drops.values):
            ax.text(c, r, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "black")
        ax.set_title("mAP50 하락률 (진할수록 많이 떨어짐)"); fig.colorbar(im, ax=ax, fraction=0.03)
        plt.tight_layout(); plt.show()
"""),
    md(r"""## 10. 크기별 AP와 FP16

- **크기별 AP**: 포트홀 대부분이 작습니다. 해상도를 낮추거나(E2) 잘라내면(E3) 작은 포트홀이 먼저 손해를 보는지, P2(E4)가 작은 포트홀을 살리는지 봅니다.
- **FP16**: 같은 가중치를 16비트로 계산 → 정확도는 거의 그대로(±0.005 이내면 사실상 동일), GPU 시간은 줄어듭니다.
"""),
    code(r"""if res is not None:
    a = res[(res["testset"] == "A") & (res["precision"] == "fp32")]
    size = a.groupby("config")[["AP_small", "AP_medium", "AP_large"]].mean().sort_index(key=agg.order_key)
    n = a.groupby("config")[["n_small", "n_medium", "n_large"]].first().iloc[0]
    ax = size.plot.bar(figsize=(11, 4), width=0.8)
    ax.set_ylabel("AP50-95 (A)"); ax.set_xlabel("")
    ax.set_title(f"크기별 AP — A의 정답 수: small {n['n_small']}, medium {n['n_medium']}, large {n['n_large']}")
    ax.grid(alpha=0.3, axis="y"); plt.xticks(rotation=30, ha="right", fontsize=8); plt.tight_layout(); plt.show()

    f = res[res["testset"] == "A"].pivot_table(index="config", columns="precision", values="mAP50_95")
    if "fp16" in f:
        f["Δ(FP16−FP32)"] = f["fp16"] - f["fp32"]
        if not spd.empty:
            g = spd[spd["device"] == "gpu"].pivot_table(index="config", columns="precision", values="total_ms")
            if {"fp32", "fp16"} <= set(g.columns):
                f["GPU 속도 향상 (×)"] = g["fp32"] / g["fp16"]
        display(f.sort_index(key=agg.order_key).round(4))
"""),
    md(r"""## 11. 예측 예시와 레이어별 시각화 (`src/eval/visualize.py`)

숫자(mAP)만으로는 모델이 **실제로 어떤 박스를 그리는지** 알기 어렵습니다. 두 종류의 그림을 만듭니다.
모든 칸은 모델에 실제로 들어가는 **640×640 정사각형 입력**(긴 변을 640에 맞추고 회색으로 채움)입니다.

**① 모델별 예측 비교 (`fig_pred_grid`)** — 행 = RDD 검증 이미지(나라마다 1장), 열 = 정답 + 모델(설정마다 시드 42)
- 초록 = 맞힌 포트홀(TP), 빨강 = 잘못 찾은 것(FP), 노란 점선 = 놓친 포트홀(FN). IoU 0.5 기준.
- 각 모델의 **검증셋에서 고정한 confidence 임계값**을 그대로 씁니다 (§0 규칙 2).
- crop 모델은 보지 못하는 위쪽 40%를 어둡게 칠했습니다.

**② 한 모델의 레이어별 흐름 (`fig_layers_*`)** — 이미지 한 장이 층을 지나며 어떻게 바뀌는지
- **1행 backbone**: 해상도가 1/4 → 1/32로 줄어드는 각 단계의 특징 맵 (채널 평균 활성도, 빨강 = 강하게 반응).
  앞쪽 층은 차선·모서리 같은 **선과 질감**, 뒤쪽 층은 도로·차처럼 **넓은 영역**에 반응합니다.
- **2행 neck**: 여러 해상도를 섞은 뒤 각 탐지 헤드(P2/P3/P4/P5)로 들어가는 특징 맵.
- **3행 detection heads**: 각 헤드가 칸마다 낸 **포트홀 점수(0~1)** 와, 그 헤드 혼자 낸 박스(흰색). 마지막 칸이 최종 예측.
  → 작은 포트홀은 고해상도 헤드(P2: 4px 칸, P3: 8px 칸), 큰 포트홀은 저해상도 헤드(P4, P5)가 잡는 것을 볼 수 있습니다.
  P2 브랜치(E4)가 실제로 작은 포트홀을 잡는지 여기서 확인합니다.
- RT-DETR은 구조(트랜스포머 디코더)가 달라 레이어별 그림은 YOLO 계열만 만듭니다.

이미지·모델은 `configs/eval.yaml`의 `visualize`에서 바꿉니다 (`stems`에 이미지 이름을 넣으면 그 이미지로). 약 20초 걸립니다.
"""),
    code(r"""from IPython.display import Image, display as show_img

RUN_VIS = True   # 예측 캐시(§4)가 있어야 함. 이미지·모델을 바꿨으면 다시 실행

if RUN_VIS and res is not None:
    _ = run("src.eval.visualize")
fig_dir = ROOT / ECFG["paths"]["figures"]
if (fig_dir / "fig_pred_grid.png").exists():
    show_img(Image(filename=str(fig_dir / "fig_pred_grid.png"), width=1400))
"""),
    code(r"""idx_csv = fig_dir / "layers/index.csv"
if idx_csv.exists():
    layer_index = pd.read_csv(idx_csv)
    display(layer_index)
    first_image = layer_index["image"].iloc[1 if len(layer_index) > 1 else 0]
    for f in layer_index[layer_index["image"] == first_image]["file"]:   # 같은 이미지, 모델별
        print(f); show_img(Image(filename=str(fig_dir / "layers" / f), width=1100))
"""),
    md(r"""## 12. Phase 3 완료 확인

계획된 모든 run × 테스트셋(A, B1, B2, B3, C, D) 조합에 결과가 있어야 Phase 3가 끝납니다.
누락된 run은 02 노트북(또는 `make train EXP=...`)에서 학습한 뒤 §4~§6을 다시 실행하세요.
"""),
    code(r"""if res is not None:
    miss = agg.missing(res)
    if miss.empty:
        print("✅ 모든 run × 테스트셋 결과가 있습니다. Phase 3 완료.")
    else:
        todo = miss.groupby(["exp", "run_id"]).size().rename("빠진 테스트셋 수").reset_index()
        print(f"⏳ 아직 {todo['run_id'].nunique()}개 run의 결과가 없습니다 (학습 후 평가 필요):")
        display(todo)
"""),
    md(r"""## 13. 해석할 때 주의할 점 (보고서 Limitations 후보)

- **속도는 PyTorch(eager) 기준**입니다. 실제 차량용 배포(TensorRT, INT8)에서는 절대값이 달라지고, 특히 YOLO26처럼 NMS가 없는 구조의 이점이 더 커질 수 있습니다. INT8은 이번 평가에 포함하지 않았습니다.
- **GPU는 RTX 3090 한 대**, CPU는 이 PC의 CPU입니다. 차량용 칩(Jetson 등)에서는 순위가 바뀔 수 있으므로, 절대 ms보다 **모델 간 상대 비교**로 해석합니다.
- **합성 비(B)** 는 실제 비를 흉내 낸 것일 뿐입니다. 실제 비의 영향은 C→D로 따로 봅니다 (다만 MWPD는 촬영 환경이 RDD와 달라 C 자체가 이미 도메인 차이를 포함).
- 시드 3개는 표준편차를 *대략* 추정하는 정도입니다. 그래서 "평균 차이 < 표준편차 합 → n.s."라는 보수적인 규칙을 씁니다.
"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3.14 (deep_learning)", "language": "python",
                   "name": "deep_learning"},
    "language_info": {"name": "python"},
}
out = Path(__file__).with_name("03_phase3_evaluation.ipynb")
nbf.write(nb, out)
print(f"Wrote {out}")
