"""Phase 3 aggregation: seed mean +- std tables, robustness drops, efficiency table and figures.

Inputs: ``outputs/results/results.csv`` (``evaluate.py``) and ``outputs/results/speed.csv``
(``speed.py``). Outputs: ``outputs/tables/tab_*.{csv,tex}``, ``outputs/figures/fig_*.{pdf,png}``.

Statistical rule (stated in the report): a configuration whose mean differs from the anchor
(``configs/eval.yaml: anchor``) by less than the sum of the two seed standard deviations is marked
"n.s." (not significant). With fewer than two seeds on either side no mark can be given ("n<2").

Usage:
    python -m src.eval.aggregate
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.common import base_parser, load_config, project_path, setup_logging

log = logging.getLogger("aggregate")
SETS = ["A", "B1", "B2", "B3", "C", "D"]
ORDER = [  # efficiency-ladder order used in every table
    "yolov8n_rdd_640",
    "yolov8s_rdd_640",
    "yolo26n_rdd_640",
    "yolo26s_rdd_640",
    "yolo26n_rdd_480",
    "yolo26n_rdd_320",
    "yolo26n_crop60_640",
    "yolo26n-p2_rdd_640",
    "yolo26n-p2_crop60_640",
    "yolo26n_rdd_640_kd-yolo26s",
    "rtdetr-l_rdd_640",
]


def load(quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the per-run results and the speed table.

    Args:
        quick: Read the ``*_quick.csv`` files.

    Returns:
        ``(results, speed)``; ``speed`` is empty if not measured yet.
    """
    p = load_config("eval")["paths"]
    sfx = "_quick" if quick else ""
    res_f = project_path(p["results"])
    spd_f = project_path(p["speed"])
    res = pd.read_csv(res_f.with_name(res_f.stem + sfx + ".csv"))
    spd_f = spd_f.with_name(spd_f.stem + sfx + ".csv")
    spd = pd.read_csv(spd_f) if spd_f.exists() else pd.DataFrame()
    return res, spd


def order_key(configs: pd.Index) -> pd.Index:
    """Sort key placing configurations in efficiency-ladder order (unknown ones last).

    Args:
        configs: Configuration names.

    Returns:
        Integer positions.
    """
    return pd.Index([ORDER.index(c) if c in ORDER else len(ORDER) for c in configs])


def seed_stats(
    res: pd.DataFrame, metric: str, testset: str, precision: str = "fp32"
) -> pd.DataFrame:
    """Mean, std and seed count of ``metric`` per configuration on one test set.

    Args:
        res: Per-run results.
        metric: Column, e.g. ``mAP50_95``.
        testset: Test-set id.
        precision: ``fp32`` or ``fp16``.

    Returns:
        DataFrame indexed by configuration with ``mean``, ``std``, ``n``.
    """
    d = res[(res["testset"] == testset) & (res["precision"] == precision)]
    g = d.groupby("config")[metric].agg(["mean", "std", "count"]).rename(columns={"count": "n"})
    g["std"] = g["std"].fillna(0.0)
    return g.sort_index(key=order_key)


def ns_mark(stats: pd.DataFrame, anchor: str) -> pd.Series:
    """Apply the "difference < sum of stds" rule against the anchor configuration.

    Args:
        stats: Output of :func:`seed_stats`.
        anchor: Anchor configuration.

    Returns:
        ``""`` (difference exceeds the noise), ``"n.s."``, ``"n<2"`` or ``"anchor"``.
    """
    if anchor not in stats.index:
        return pd.Series("no anchor", index=stats.index)
    a = stats.loc[anchor]
    out = {}
    for c, r in stats.iterrows():
        if c == anchor:
            out[c] = "anchor"
        elif r["n"] < 2 or a["n"] < 2:
            out[c] = "n<2"
        else:
            out[c] = "n.s." if abs(r["mean"] - a["mean"]) < r["std"] + a["std"] else ""
    return pd.Series(out)


def pm(mean: float, std: float, n: int, digits: int = 3) -> str:
    """Format ``mean +- std`` (or just the mean for a single seed).

    Args:
        mean: Mean.
        std: Standard deviation.
        n: Seeds.
        digits: Decimals.

    Returns:
        Formatted string.
    """
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if n > 1 else f"{mean:.{digits}f}"


def accuracy_table(res: pd.DataFrame, anchor: str) -> pd.DataFrame:
    """Test set A accuracy per configuration (mean +- std over seeds) with n.s. marks.

    Args:
        res: Per-run results.
        anchor: Anchor configuration.

    Returns:
        Display table.
    """
    cols = {"mAP50": "mAP50", "mAP50_95": "mAP50-95", "F1": "F1", "AP_small": "AP_small"}
    out = pd.DataFrame()
    for m, name in cols.items():
        s = seed_stats(res, m, "A")
        out[name] = [pm(r["mean"], r["std"], int(r["n"])) for _, r in s.iterrows()]
        out.index = s.index
        if m == "mAP50_95":
            out["seeds"] = s["n"].astype(int)
            out["vs anchor (mAP50-95)"] = ns_mark(s, anchor)
    return out


def robustness_table(res: pd.DataFrame) -> pd.DataFrame:
    """Mean mAP50 on every test set plus relative drops.

    ``drop(X) = (mAP50_A - mAP50_X) / mAP50_A`` for B1-B3 and C; the real-rain drop is C -> D.

    Args:
        res: Per-run results.

    Returns:
        Table indexed by configuration.
    """
    d = res[res["precision"] == "fp32"]
    per_run = d.pivot_table(index=["config", "run_id"], columns="testset", values="mAP50")
    for x in ["B1", "B2", "B3", "C"]:
        if x in per_run and "A" in per_run:
            per_run[f"drop_{x}"] = (per_run["A"] - per_run[x]) / per_run["A"]
    if "C" in per_run and "D" in per_run:
        per_run["drop_C→D"] = (per_run["C"] - per_run["D"]) / per_run["C"]
    t = per_run.groupby("config").mean()
    cols = [c for c in SETS if c in t] + [c for c in t if c.startswith("drop")]
    return t[cols].sort_index(key=order_key)


def efficiency_table(res: pd.DataFrame, spd: pd.DataFrame) -> pd.DataFrame:
    """Cost (params, effective GFLOPs, latency) next to accuracy, incl. the FP16 accuracy change.

    Args:
        res: Per-run results.
        spd: Speed table.

    Returns:
        Table indexed by configuration.
    """
    a32 = seed_stats(res, "mAP50_95", "A")
    t = pd.DataFrame({"mAP50-95 (A)": a32["mean"]})
    t["params_M"] = res.groupby("config")["params_M"].first()
    fp16 = res[(res["testset"] == "A") & (res["precision"] == "fp16")]
    if not fp16.empty:
        f = fp16.groupby("config")["mAP50_95"].mean()
        t["ΔmAP50-95 FP16"] = f - t["mAP50-95 (A)"]
    if not spd.empty:
        t["GFLOPs_eff"] = spd.groupby("config")["gflops_eff"].first()
        for (dev, prec), g in spd.groupby(["device", "precision"]):
            t[f"{dev.upper()} {prec.upper()} ms"] = g.set_index("config")["total_ms"]
    return t.sort_index(key=order_key)


def missing(res: pd.DataFrame) -> pd.DataFrame:
    """Planned run x test-set combinations that have no result yet.

    Args:
        res: Per-run results.

    Returns:
        ``run_id``, ``exp``, ``testset`` rows (empty when complete).
    """
    from src.train.train import plan_runs

    exps = list(load_config("experiments")["experiments"])
    have = set(zip(res["run_id"], res["testset"], strict=True))
    rows = [
        {"run_id": r.run_id, "exp": r.exp, "testset": s}
        for r in plan_runs(exps, None, None)
        for s in SETS
        if (r.run_id, s) not in have
    ]
    return pd.DataFrame(rows, columns=["run_id", "exp", "testset"])


def save(df: pd.DataFrame, name: str, tables: Path, floatfmt: str = "%.3f") -> None:
    """Write a table as CSV and LaTeX (booktabs).

    Args:
        df: Table.
        name: File stem.
        tables: Output folder.
        floatfmt: Float format for LaTeX.
    """
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / f"{name}.csv")
    (tables / f"{name}.tex").write_text(
        df.to_latex(float_format=floatfmt, escape=True, na_rep="--"), encoding="utf-8"
    )


def fig_pareto(eff: pd.DataFrame, out: Path | None = None) -> plt.Figure:
    """Accuracy vs cost: mAP50-95 on A against GFLOPs, GPU FP16 latency and CPU latency.

    Args:
        eff: Output of :func:`efficiency_table`.
        out: Save path stem (``.pdf`` and ``.png``), or ``None``.

    Returns:
        Figure.
    """
    xs = [c for c in ["GFLOPs_eff", "GPU FP16 ms", "CPU FP32 ms"] if c in eff] or ["params_M"]
    fig, axes = plt.subplots(1, len(xs), figsize=(5 * len(xs), 4), squeeze=False)
    for ax, x in zip(axes[0], xs, strict=True):
        d = eff.dropna(subset=[x, "mAP50-95 (A)"]).sort_values(x)
        ax.scatter(d[x], d["mAP50-95 (A)"], s=40, zorder=3)
        for c, r in d.iterrows():
            ax.annotate(
                c, (r[x], r["mAP50-95 (A)"]), fontsize=7, xytext=(3, 3), textcoords="offset points"
            )
        front, best = [], -np.inf  # Pareto front: cheaper and at least as accurate
        for _, r in d.iterrows():
            if r["mAP50-95 (A)"] > best:
                front.append((r[x], r["mAP50-95 (A)"]))
                best = r["mAP50-95 (A)"]
        if front:
            ax.step(
                *zip(*front, strict=True),
                where="post",
                color="tab:red",
                lw=1,
                alpha=0.6,
                label="Pareto front",
            )
        ax.set_xlabel(x.replace("_eff", " (actual input)"))
        ax.set_ylabel("mAP50-95 on A")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    if out:
        for ext in ("pdf", "png"):
            fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    return fig


def fig_robustness(rob: pd.DataFrame, out: Path | None = None) -> plt.Figure:
    """mAP50 across rain levels (A -> B1 -> B2 -> B3) with C and D as separate markers.

    Args:
        rob: Output of :func:`robustness_table`.
        out: Save path stem, or ``None``.

    Returns:
        Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    levels = [c for c in ["A", "B1", "B2", "B3"] if c in rob]
    for i, (c, r) in enumerate(rob.iterrows()):
        color = plt.cm.tab20(i % 20)
        ax.plot(range(len(levels)), r[levels], marker="o", color=color, label=c, lw=1.2)
        for j, s in enumerate([s for s in ["C", "D"] if s in rob]):
            ax.scatter(len(levels) + 0.5 + j, r[s], color=color, marker="s^"[j], s=30)
    ticks = ["clear (A)", "drizzle (B1)", "heavy (B2)", "torrential (B3)"][: len(levels)]
    extra = [s for s in ["MWPD normal (C)", "MWPD rain (D)"] if s[-2] in rob]
    ax.set_xticks(list(range(len(levels))) + [len(levels) + 0.5 + j for j in range(len(extra))])
    ax.set_xticklabels(ticks + extra, rotation=20, fontsize=8)
    ax.set_ylabel("mAP50")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    if out:
        for ext in ("pdf", "png"):
            fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    return fig


def main() -> None:
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    args = parser.parse_args()
    setup_logging("aggregate")
    ecfg = load_config("eval")
    res, spd = load(args.quick)
    tables = project_path(ecfg["paths"]["tables"])
    figs = project_path(ecfg["paths"]["figures"])
    figs.mkdir(parents=True, exist_ok=True)
    sfx = "_quick" if args.quick else ""
    acc = accuracy_table(res, ecfg["anchor"])
    rob = robustness_table(res)
    eff = efficiency_table(res, spd)
    log.info("Accuracy on A:\n%s", acc.to_string())
    log.info("Robustness (mAP50, drops):\n%s", rob.round(3).to_string())
    log.info("Efficiency:\n%s", eff.round(3).to_string())
    if args.dry_run:
        return
    save(acc, f"tab_accuracy{sfx}", tables)
    save(rob, f"tab_robustness{sfx}", tables)
    save(eff, f"tab_efficiency{sfx}", tables)
    fig_pareto(eff, figs / f"fig_pareto{sfx}")
    fig_robustness(rob, figs / f"fig_robustness{sfx}")
    miss = missing(res)
    miss.to_csv(project_path(ecfg["paths"]["results"]).with_name(f"missing{sfx}.csv"), index=False)
    if miss.empty:
        log.info("Phase 3 complete: every planned run x test set has a result")
    else:
        log.warning(
            "%d run x test-set results missing (%d runs), see missing%s.csv:\n%s",
            len(miss),
            miss["run_id"].nunique(),
            sfx,
            miss.groupby(["exp", "run_id"]).size().to_string(),
        )


if __name__ == "__main__":
    main()
