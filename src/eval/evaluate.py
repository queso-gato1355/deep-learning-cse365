"""Phase 3 accuracy evaluation: every finished run x every test set (``configs/eval.yaml``).

For each run in ``outputs/runs/*/run_meta.json``:
    1. Predict on the RDD val list and fix the confidence threshold that maximises F1 there
       (the threshold is never tuned on a test set).
    2. Predict on test sets A, B1, B2, B3, C, D (and A again with FP16 weights).
    3. Score mAP50, mAP50-95 (Ultralytics matching + AP code), precision/recall/F1 at the fixed
       threshold, and COCO small/medium/large AP (``pycocotools``).

Road-crop runs (``data_key`` ``cropNN``) get the bottom NN% of each full frame, exactly as in a car,
and their boxes are shifted back to full-frame coordinates, so potholes outside the crop are misses.

Predictions are cached in ``outputs/eval/preds/{run_id}/{set}.npz``; re-runs only score.
One row per run x set is written to ``outputs/results/results.csv``.

Usage:
    python -m src.eval.evaluate                  # all finished runs, all sets
    python -m src.eval.evaluate --runs yolo26n_rdd_640_s42 --sets A C
    python -m src.eval.evaluate --quick          # 50 images per set, own cache (pipeline check)
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.synth_rain import read_list
from src.utils.common import base_parser, imread, load_config, project_path, setup_logging

log = logging.getLogger("evaluate")
IOUV = np.linspace(0.5, 0.95, 10)
QUICK_IMAGES = 50


def crop_keep(data_key: str) -> float | None:
    """Return the kept height fraction for a road-crop dataset key (``crop60`` -> 0.6).

    Args:
        data_key: Dataset key from ``run_meta.json``.

    Returns:
        Fraction, or ``None`` for full-frame runs.
    """
    return int(data_key[4:]) / 100 if data_key.startswith("crop") else None


def label_path(img: Path) -> Path:
    """Map ``.../images/x.jpg`` to ``.../labels/x.txt`` (last ``images`` folder only).

    Args:
        img: Image path.

    Returns:
        Label path.
    """
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_gt(img: Path, h: int, w: int) -> np.ndarray:
    """Read YOLO labels as pixel ``xyxy`` boxes.

    Args:
        img: Image path.
        h: Image height.
        w: Image width.

    Returns:
        ``(n, 4)`` float32 array.
    """
    rows = [
        r.split() for r in label_path(img).read_text(encoding="utf-8").splitlines() if r.strip()
    ]
    if not rows:
        return np.zeros((0, 4), np.float32)
    a = np.array(rows, np.float32)[:, 1:5]
    cx, cy, bw, bh = a[:, 0] * w, a[:, 1] * h, a[:, 2] * w, a[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def predict_set(
    model: Any, images: list[Path], imgsz: int, keep: float | None, half: bool, pcfg: dict
) -> dict[str, np.ndarray]:
    """Run the model over ``images`` and collect predictions and ground truth.

    Args:
        model: Ultralytics model.
        images: Image paths.
        imgsz: Inference size (same as training).
        keep: Road-crop fraction, or ``None``.
        half: FP16 inference.
        pcfg: ``predict`` section of the eval config.

    Returns:
        Flat arrays: ``pred_img, pred_box, pred_conf, gt_img, gt_box, hw``.
    """
    out: dict[str, list] = {
        k: [] for k in ("pred_img", "pred_box", "pred_conf", "gt_img", "gt_box")
    }
    hw = []
    bs = pcfg["batch"]
    for s in tqdm(range(0, len(images), bs), desc="predict", leave=False):
        chunk = images[s : s + bs]
        ims, offs = [], []
        for i, p in enumerate(chunk, start=s):
            im = imread(p)
            h, w = im.shape[:2]
            hw.append((h, w))
            gt = read_gt(p, h, w)
            out["gt_img"].append(np.full(len(gt), i, np.int32))
            out["gt_box"].append(gt)
            y0 = int(round(h * (1 - keep))) if keep else 0
            ims.append(im[y0:])
            offs.append(y0)
        res = model.predict(
            ims,
            imgsz=imgsz,
            conf=pcfg["conf"],
            iou=pcfg["iou"],
            max_det=pcfg["max_det"],
            quantize=16 if half else None,
            verbose=False,
        )
        for i, (r, y0) in enumerate(zip(res, offs, strict=True), start=s):
            b = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            b[:, [1, 3]] += y0
            out["pred_img"].append(np.full(len(b), i, np.int32))
            out["pred_box"].append(b)
            out["pred_conf"].append(r.boxes.conf.cpu().numpy().astype(np.float32))
    arrs = {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}
    arrs["pred_box"] = arrs["pred_box"].reshape(-1, 4)
    arrs["gt_box"] = arrs["gt_box"].reshape(-1, 4)
    arrs["hw"] = np.array(hw, np.int32)
    return arrs


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of ``xyxy`` boxes.

    Args:
        a: ``(m, 4)``.
        b: ``(n, 4)``.

    Returns:
        ``(m, n)`` IoU matrix.
    """
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def match(p: dict[str, np.ndarray]) -> np.ndarray:
    """Mark each prediction correct/incorrect at IoU 0.50:0.95 (Ultralytics greedy matching).

    Args:
        p: Arrays from :func:`predict_set`.

    Returns:
        ``(n_pred, 10)`` bool array aligned with ``p["pred_conf"]``.
    """
    correct = np.zeros((len(p["pred_conf"]), len(IOUV)), bool)
    n_img = len(p["hw"])
    for i in range(n_img):
        pi = np.flatnonzero(p["pred_img"] == i)
        gi = np.flatnonzero(p["gt_img"] == i)
        if not len(pi) or not len(gi):
            continue
        pi = pi[np.argsort(-p["pred_conf"][pi])]
        iou = box_iou(p["gt_box"][gi], p["pred_box"][pi])
        matched = np.zeros((len(gi), len(IOUV)), bool)
        for j in np.flatnonzero((iou >= IOUV[0]).any(0)):
            avail = np.where(matched, 0, iou[:, j, None])
            k = avail.argmax(0)
            ok = avail[k, range(len(IOUV))] >= IOUV
            correct[pi[j]] = ok
            matched[k, range(len(IOUV))] |= ok
    return correct


def score(p: dict[str, np.ndarray], conf_thr: float | None) -> dict[str, float]:
    """Compute mAP and P/R/F1 (at ``conf_thr``, or at the F1-maximising threshold if ``None``).

    Args:
        p: Arrays from :func:`predict_set`.
        conf_thr: Fixed confidence threshold.

    Returns:
        Metric dictionary (``conf_thr`` included).
    """
    from ultralytics.utils.metrics import ap_per_class

    tp = match(p)
    conf = p["pred_conf"]
    n_gt = len(p["gt_box"])
    res: dict[str, float] = {"n_img": len(p["hw"]), "n_gt": n_gt, "n_pred": len(conf)}
    if n_gt == 0 or len(conf) == 0:
        return res | {"mAP50": 0.0, "mAP50_95": 0.0, "P": 0.0, "R": 0.0, "F1": 0.0}
    out = ap_per_class(tp, conf, np.zeros(len(conf)), np.zeros(n_gt))
    ap, f1_curve, x = out[5], out[9], out[10]
    if conf_thr is None:
        conf_thr = float(x[f1_curve[0].argmax()])
    keep = conf >= conf_thr
    tpk = tp[keep, 0].sum()
    prec = tpk / max(keep.sum(), 1)
    rec = tpk / n_gt
    f1 = 2 * prec * rec / max(prec + rec, 1e-16)
    return res | {
        "mAP50": float(ap[0, 0]),
        "mAP50_95": float(ap[0].mean()),
        "P": float(prec),
        "R": float(rec),
        "F1": float(f1),
        "conf_thr": conf_thr,
    }


def size_ap(p: dict[str, np.ndarray]) -> dict[str, float]:
    """COCO AP50-95 for small/medium/large boxes (areas < 32^2, < 96^2, rest; original pixels).

    Args:
        p: Arrays from :func:`predict_set`.

    Returns:
        ``AP_small``, ``AP_medium``, ``AP_large`` (``nan`` when a size has no ground truth) and
        the ground-truth count per size.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = p["gt_box"]
    wh = gt[:, 2:] - gt[:, :2]
    area = wh.prod(1)
    counts = {
        "n_small": int((area < 32**2).sum()),
        "n_medium": int(((area >= 32**2) & (area < 96**2)).sum()),
        "n_large": int((area >= 96**2).sum()),
    }
    if len(gt) == 0 or len(p["pred_conf"]) == 0:
        return counts | {"AP_small": np.nan, "AP_medium": np.nan, "AP_large": np.nan}
    images = [{"id": i, "height": int(h), "width": int(w)} for i, (h, w) in enumerate(p["hw"])]
    anns = [
        {
            "id": k + 1,
            "image_id": int(i),
            "category_id": 1,
            "bbox": [float(b[0]), float(b[1]), float(s[0]), float(s[1])],
            "area": float(a),
            "iscrowd": 0,
        }
        for k, (i, b, s, a) in enumerate(zip(p["gt_img"], gt, wh, area, strict=True))
    ]
    pw = p["pred_box"][:, 2:] - p["pred_box"][:, :2]
    dets = [
        {
            "image_id": int(i),
            "category_id": 1,
            "bbox": [float(b[0]), float(b[1]), float(s[0]), float(s[1])],
            "score": float(c),
        }
        for i, b, s, c in zip(p["pred_img"], p["pred_box"], pw, p["pred_conf"], strict=True)
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1}]}
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes(dets), "bbox")
        ev.params.maxDets = [1, 10, 300]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = [float(v) if v >= 0 else np.nan for v in ev.stats[3:6]]
    return counts | {"AP_small": s[0], "AP_medium": s[1], "AP_large": s[2]}


def load_runs(runs_dir: Path, only: list[str] | None) -> list[dict[str, Any]]:
    """Read ``run_meta.json`` of every finished run.

    Args:
        runs_dir: ``outputs/runs``.
        only: Optional run-id filter.

    Returns:
        Run metadata dictionaries sorted by run id.
    """
    metas = []
    for f in sorted(runs_dir.glob("*/run_meta.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if only and m["run_id"] not in only:
            continue
        if not (f.parent / "weights" / "best.pt").exists():
            log.warning("[%s] best.pt missing - skipped", m["run_id"])
            continue
        metas.append(m)
    return metas


def cached_predict(
    meta: dict, name: str, images: list[Path], half: bool, cache_dir: Path, force: bool, state: dict
) -> dict[str, np.ndarray]:
    """Load cached predictions or compute and cache them (the model is loaded lazily).

    Args:
        meta: Run metadata.
        name: Cache name (test-set id, ``val`` or ``A_fp16``).
        images: Image paths.
        half: FP16 inference.
        cache_dir: Per-run cache folder.
        force: Ignore the cache.
        state: Holds the loaded model across calls.

    Returns:
        Prediction arrays.
    """
    f = cache_dir / f"{name}.npz"
    if f.exists() and not force:
        return dict(np.load(f))
    if "model" not in state:
        from ultralytics import RTDETR, YOLO

        cls = RTDETR if "rtdetr" in meta["cfg"] else YOLO
        state["model"] = cls(str(project_path("outputs/runs") / meta["run_id"] / "weights/best.pt"))
    p = predict_set(
        state["model"], images, meta["imgsz"], crop_keep(meta["data_key"]), half, state["pcfg"]
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **p)
    return p


def evaluate_run(
    meta: dict, ecfg: dict, sets: list[str], quick: bool, force: bool
) -> list[dict[str, Any]]:
    """Evaluate one run on the val list (threshold) and the requested test sets.

    Args:
        meta: Run metadata.
        ecfg: Eval config.
        sets: Test-set ids.
        quick: Use only the first images of each list.
        force: Recompute cached predictions.

    Returns:
        One result row per (test set, precision).
    """
    root = project_path(ecfg["paths"]["preds"] + ("_quick" if quick else ""))
    cache = root / meta["run_id"]
    state: dict[str, Any] = {"pcfg": ecfg["predict"]}

    def imgs(rel: str) -> list[Path]:
        lst = read_list(project_path(rel))
        return lst[:QUICK_IMAGES] if quick else lst

    val = cached_predict(meta, "val", imgs(ecfg["val_list"]), False, cache, force, state)
    thr = score(val, None)["conf_thr"]
    base = {
        k: meta.get(k)
        for k in ("run_id", "exp", "model", "data_key", "imgsz", "seed", "teacher_run")
    }
    base |= {"config": meta["run_id"].rsplit("_s", 1)[0], "params_M": meta.get("params_M")}
    base |= {"gflops_train_input": meta.get("gflops"), "conf_thr": thr}
    jobs = [(s, "fp32") for s in sets] + [(s, "fp16") for s in ecfg["fp16_testsets"] if s in sets]
    rows = []
    for s, prec in jobs:
        name = s if prec == "fp32" else f"{s}_fp16"
        p = cached_predict(
            meta, name, imgs(ecfg["testsets"][s]), prec == "fp16", cache, force, state
        )
        m = score(p, thr) | size_ap(p)
        rows.append(base | {"testset": s, "precision": prec} | m)
        log.info(
            "[%s] %-3s %s  mAP50 %.3f  mAP50-95 %.3f  F1 %.3f",
            meta["run_id"],
            s,
            prec,
            m["mAP50"],
            m["mAP50_95"],
            m["F1"],
        )
    if "model" in state:
        del state["model"]
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    parser.add_argument("--runs", nargs="+", help="Only these run ids")
    parser.add_argument("--sets", nargs="+", help="Only these test sets (default: all)")
    args = parser.parse_args()
    setup_logging("evaluate")
    ecfg = load_config("eval")
    sets = args.sets or list(ecfg["testsets"])
    metas = load_runs(project_path("outputs/runs"), args.runs)
    log.info("%d finished runs x %d test sets: %s", len(metas), len(sets), ", ".join(sets))
    if args.dry_run:
        for m in metas:
            log.info("  %s", m["run_id"])
        return
    out = project_path(ecfg["paths"]["results"])
    if args.quick:
        out = out.with_name(out.stem + "_quick.csv")
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    rows = []
    for m in tqdm(metas, desc="runs"):
        rows += evaluate_run(m, ecfg, sets, args.quick, args.force)
    new = pd.DataFrame(rows)
    if not old.empty:  # keep rows of runs/sets not evaluated this time
        key = ["run_id", "testset", "precision"]
        old = old.merge(new[key], on=key, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new = pd.concat([old, new], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(["run_id", "testset", "precision"]).to_csv(out, index=False)
    log.info("Wrote %d rows to %s", len(new), out)


if __name__ == "__main__":
    main()
