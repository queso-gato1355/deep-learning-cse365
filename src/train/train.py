"""Train the Phase 2 efficiency ladder defined in ``configs/experiments.yaml``.

Usage:
    python -m src.train.train --exp E1 --list            # show runs + time estimate, train nothing
    python -m src.train.train --exp E1 --yes             # train (finished runs are skipped)
    python -m src.train.train --exp E1 --quick --yes     # 5% data, 3 epochs, outputs/runs_quick
    python -m src.train.train --exp E1 --models yolo26n --seeds 42 --yes
    python -m src.train.train --exp E1 --finalize        # write run_meta.json for runs whose
                                                         # training finished but the process died

Weights land in ``outputs/runs/{run_id}/weights/best.pt`` plus a ``run_meta.json`` summary.
On Google Colab, ``colab_overrides`` from ``configs/train.yaml`` replace the matching settings.
Pretrained weights are downloaded once into ``outputs/weights/`` (git-ignored).
``--offline`` skips the weight and AMP-check downloads and disables plots: use it only to test the
pipeline without internet; such runs start from random weights. (Ultralytics may still fetch its
plot font ``Arial.ttf``, 755 KB, once into its settings folder.)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.utils.common import base_parser, load_config, project_path, setup_logging
from src.utils.flops import count_gflops, count_params

log = logging.getLogger("train")


@dataclass
class Run:
    """One training run of the experiment matrix."""

    run_id: str
    exp: str
    model: str
    cfg: str
    weights: str | None
    data_key: str
    data_yaml: str
    imgsz: int
    seed: int
    teacher_run: str | None = None


def as_list(v: Any) -> list:
    """Wrap a scalar in a list.

    Args:
        v: Scalar or list.

    Returns:
        List.
    """
    return v if isinstance(v, list) else [v]


def plan_runs(exp_ids: list[str], models: list[str] | None, seeds: list[int] | None) -> list[Run]:
    """Expand experiments into concrete runs.

    Args:
        exp_ids: Experiment ids (e.g. ``["E1", "E3"]``).
        models: Optional model-name filter.
        seeds: Optional seed filter.

    Returns:
        Runs in execution order (duplicates removed).
    """
    cfg = load_config("experiments")
    runs: dict[str, Run] = {}
    for e in exp_ids:
        spec = cfg["experiments"][e]
        for m in spec["models"]:
            if models and m not in models:
                continue
            for d in as_list(spec["data"]):
                for sz in as_list(spec["imgsz"]):
                    for s in spec["seeds"]:
                        if seeds and s not in seeds:
                            continue
                        variant = f"{d}_{sz}"
                        teacher = None
                        if spec.get("teacher"):
                            teacher = f"{spec['teacher']}_{variant}_s{s}"
                            variant += f"_kd-{spec['teacher']}"
                        run_id = f"{m}_{variant}_s{s}"
                        runs[run_id] = Run(
                            run_id=run_id,
                            exp=e,
                            model=m,
                            cfg=cfg["models"][m]["cfg"],
                            weights=cfg["models"][m].get("weights"),
                            data_key=d,
                            data_yaml=str(project_path(cfg["datasets"][d])),
                            imgsz=sz,
                            seed=s,
                            teacher_run=teacher,
                        )
    return list(runs.values())


def model_cost(cfg_name: str, imgsz: int) -> tuple[float, float]:
    """Params (M) and GFLOPs of a fresh 1-class model at a square input.

    Args:
        cfg_name: Ultralytics model YAML.
        imgsz: Input side.

    Returns:
        ``(params_M, gflops)``.
    """
    from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel
    from ultralytics.utils import LOGGER

    cls = RTDETRDetectionModel if "rtdetr" in cfg_name else DetectionModel
    level = LOGGER.level
    LOGGER.setLevel(logging.WARNING)  # hide "Overriding model.yaml nc=80 with nc=1"
    m = cls(cfg_name, nc=1, verbose=False).eval()
    LOGGER.setLevel(level)
    return count_params(m), count_gflops(m, (imgsz, imgsz))


def estimate_hours(runs: list[Run], runs_dir: Path, epochs: int, fallback: float) -> pd.DataFrame:
    """Estimate wall time per run from finished runs' measured speed (or a fallback constant).

    Args:
        runs: Planned runs.
        runs_dir: Directory with finished runs (``run_meta.json``).
        epochs: Maximum epochs per run.
        fallback: Seconds per epoch per training GFLOP when nothing has finished yet.

    Returns:
        Table with status, cost and estimated hours (upper bound: early stopping cuts it).
    """
    measured = []
    for meta in runs_dir.glob("*/run_meta.json"):
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("sec_per_epoch") and m.get("gflops") and not m.get("teacher_run"):
            measured.append(m["sec_per_epoch"] / m["gflops"])
    k = float(pd.Series(measured).median()) if measured else fallback
    rows, cache = [], {}
    for r in runs:
        key = (r.cfg, r.imgsz)
        if key not in cache:
            cache[key] = model_cost(r.cfg, r.imgsz)
        params, gf = cache[key]
        factor = 1.5 if r.teacher_run else 1.0  # teacher forward pass during distillation
        done = (runs_dir / r.run_id / "run_meta.json").exists()
        rows.append(
            {
                "run_id": r.run_id,
                "exp": r.exp,
                "params_M": round(params, 2),
                "GFLOPs": round(gf, 2),
                "status": "done" if done else "todo",
                "est_hours": 0.0 if done else round(k * gf * factor * epochs / 3600, 2),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["k"] = k
    df.attrs["source"] = f"{len(measured)} finished runs" if measured else "fallback constant"
    return df


def transferred_items(cfg_name: str, weights: str) -> tuple[int, int]:
    """Count how many tensors of a 1-class model can be loaded from pretrained weights.

    Args:
        cfg_name: Model YAML.
        weights: Pretrained checkpoint (downloaded by Ultralytics if missing).

    Returns:
        ``(loaded, total)`` state-dict entries matched by name and shape.
    """
    from ultralytics.nn.tasks import DetectionModel, load_checkpoint

    model = DetectionModel(cfg_name, nc=1, verbose=False)
    ckpt, _ = load_checkpoint(weights)
    src = ckpt.float().state_dict()
    dst = model.state_dict()
    loaded = sum(1 for k, v in dst.items() if k in src and src[k].shape == v.shape)
    return loaded, len(dst)


def train_one(r: Run, tcfg: dict, runs_dir: Path, quick: bool, offline: bool) -> dict:
    """Train one run with Ultralytics and write ``run_meta.json``.

    Args:
        r: Run spec.
        tcfg: ``configs/train.yaml``.
        runs_dir: Output root.
        quick: Pipeline-check mode.
        offline: Skip all downloads (random init, no AMP check, no plots).

    Returns:
        The run summary written to ``run_meta.json``.
    """
    from ultralytics import RTDETR, YOLO, settings

    settings.update({"sync": False})  # no analytics / crash reports sent to Ultralytics
    model_cls = RTDETR if "rtdetr" in r.cfg else YOLO
    model = model_cls(r.cfg)
    loaded = total = None
    if r.weights and not offline:
        loaded, total = transferred_items(r.cfg, r.weights)
        log.info(
            "[%s] pretrained %s: %d/%d tensors match (rest start random)",
            r.run_id,
            r.weights,
            loaded,
            total,
        )
        model.load(r.weights)
    teacher = None
    if r.teacher_run:
        teacher = runs_dir / r.teacher_run / "weights" / "best.pt"
        if not teacher.exists():
            raise FileNotFoundError(f"Teacher {teacher} missing: run its experiment (E1) first")
    q = tcfg["quick"]
    args = {
        "data": r.data_yaml,
        "imgsz": r.imgsz,
        "epochs": q["epochs"] if quick else tcfg["epochs"],
        "fraction": q["fraction"] if quick else 1.0,
        "patience": tcfg["patience"],
        "batch": tcfg["batch"],
        "optimizer": tcfg["optimizer"],
        "cos_lr": tcfg["cos_lr"],
        "workers": tcfg["workers"],
        "device": tcfg["device"] if torch.cuda.is_available() else "cpu",
        "cache": tcfg["cache"],
        "deterministic": tcfg["deterministic"],
        "seed": r.seed,
        "project": str(runs_dir),
        "name": r.run_id,
        "exist_ok": True,
        "pretrained": bool(r.weights) and not offline,
        "amp": not offline,
        "plots": not offline,
    }
    if teacher is not None:
        args |= {"distill_model": str(teacher), "dis": tcfg["distill_weight"]}
    t0 = time.time()
    model.train(**args)
    wall = time.time() - t0
    pretrained = f"{loaded}/{total}" if loaded is not None else None
    return write_meta(r, runs_dir / r.run_id, wall, pretrained, quick, offline)


def write_meta(
    r: Run,
    run_dir: Path,
    wall: float,
    pretrained: str | None,
    quick: bool,
    offline: bool,
    extra: dict | None = None,
) -> dict:
    """Summarize a run from its ``results.csv`` and write ``run_meta.json``.

    Args:
        r: Run spec.
        run_dir: The run's output folder.
        wall: Training wall time in seconds.
        pretrained: ``"loaded/total"`` pretrained tensors, or None.
        quick: Pipeline-check mode.
        offline: Run without downloads.
        extra: Additional fields to record.

    Returns:
        The summary.
    """
    res = pd.read_csv(run_dir / "results.csv")
    res.columns = [c.strip() for c in res.columns]
    best = res.loc[res["metrics/mAP50-95(B)"].idxmax()]
    params, gf = model_cost(r.cfg, r.imgsz)
    meta = {
        **asdict(r),
        "quick": quick,
        "offline": offline,
        "pretrained_tensors": pretrained,
        "params_M": round(params, 3),
        "gflops": round(gf, 3),
        "epochs_done": len(res),
        "sec_per_epoch": round(wall / max(len(res), 1), 1),
        "wall_hours": round(wall / 3600, 3),
        "best_epoch": int(best["epoch"]),
        "val_mAP50": round(float(best["metrics/mAP50(B)"]), 4),
        "val_mAP50_95": round(float(best["metrics/mAP50-95(B)"]), 4),
        "val_precision": round(float(best["metrics/precision(B)"]), 4),
        "val_recall": round(float(best["metrics/recall(B)"]), 4),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "finished": datetime.now().isoformat(timespec="seconds"),
        **(extra or {}),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def finalize_one(r: Run, runs_dir: Path, quick: bool) -> dict | None:
    """Write ``run_meta.json`` for a run whose training finished but whose process died after.

    Ultralytics saves ``best.pt`` and ``results.csv`` before its final validation, so a run killed
    there (e.g. out of RAM) has everything the summary needs. The summary comes from
    ``results.csv`` exactly as after a normal run; only the wall time is taken from the
    ``time`` column (cumulative seconds) instead of being measured.

    Args:
        r: Run spec.
        runs_dir: Output root.
        quick: Pipeline-check mode.

    Returns:
        The summary, or None if the run has no ``best.pt``/``results.csv``.
    """
    run_dir = runs_dir / r.run_id
    if not ((run_dir / "weights" / "best.pt").exists() and (run_dir / "results.csv").exists()):
        return None
    res = pd.read_csv(run_dir / "results.csv")
    res.columns = [c.strip() for c in res.columns]
    wall = float(res["time"].iloc[-1]) if "time" in res.columns else 0.0
    pretrained = None
    if r.weights:
        loaded, total = transferred_items(r.cfg, r.weights)
        pretrained = f"{loaded}/{total}"
    return write_meta(r, run_dir, wall, pretrained, quick, False, {"finalized_after_crash": True})


def main() -> None:
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    parser.add_argument("--exp", nargs="+", required=True, help="Experiment ids, e.g. E1 E3")
    parser.add_argument("--models", nargs="+", help="Only these models")
    parser.add_argument("--seeds", nargs="+", type=int, help="Only these seeds")
    parser.add_argument("--list", action="store_true", help="Print the plan and time estimate only")
    parser.add_argument("--yes", action="store_true", help="Confirm: actually start training")
    parser.add_argument("--offline", action="store_true", help="No downloads (random init)")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Train nothing; write run_meta.json for runs that have best.pt but no summary",
    )
    args = parser.parse_args()
    setup_logging("train")
    tcfg = load_config("train")
    if "COLAB_RELEASE_TAG" in os.environ and tcfg.get("colab_overrides"):
        tcfg |= tcfg["colab_overrides"]
        log.info("Google Colab detected: using %s", tcfg["colab_overrides"])
    runs_dir = project_path("outputs/runs_quick" if args.quick else "outputs/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    runs = plan_runs(args.exp, args.models, args.seeds)
    epochs = tcfg["quick"]["epochs"] if args.quick else tcfg["epochs"]
    table = estimate_hours(runs, runs_dir, epochs, tcfg["sec_per_epoch_per_gflop"])
    log.info(
        "Plan (%d runs, %d to do), max %d epochs each, estimate from %s:\n%s",
        len(runs),
        (table["status"] == "todo").sum(),
        epochs,
        table.attrs["source"],
        table.to_string(index=False),
    )
    log.info(
        "Estimated upper bound: %.1f h (early stopping usually ends sooner)",
        table["est_hours"].sum(),
    )
    if args.list or args.dry_run:
        return
    if args.finalize:
        weights_dir = project_path("outputs/weights")  # where transferred_items finds the .pt files
        weights_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(weights_dir)
        for r in runs:
            if (runs_dir / r.run_id / "run_meta.json").exists():
                continue
            meta = finalize_one(r, runs_dir, args.quick)
            if meta is None:
                log.info("[%s] no best.pt/results.csv - nothing to finalize", r.run_id)
            else:
                log.info(
                    "[%s] finalized: val mAP50-95 %.3f (best epoch %d of %d)",
                    r.run_id,
                    meta["val_mAP50_95"],
                    meta["best_epoch"],
                    meta["epochs_done"],
                )
        return
    if not args.yes:
        log.warning("Nothing trained. Re-run with --yes to start (check the plan above first).")
        return
    # Ultralytics downloads pretrained weights (and its AMP-check model) into the working directory;
    # keep them in outputs/weights instead of the project root. All other paths are absolute.
    weights_dir = project_path("outputs/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(weights_dir)
    for r in runs:
        if (runs_dir / r.run_id / "run_meta.json").exists() and not args.force:
            log.info("[%s] already finished - skipped (use --force to retrain)", r.run_id)
            continue
        log.info("[%s] training (%s)", r.run_id, r.exp)
        meta = train_one(r, tcfg, runs_dir, args.quick, args.offline)
        log.info(
            "[%s] done: val mAP50 %.3f, mAP50-95 %.3f, %d epochs, %.2f h",
            r.run_id,
            meta["val_mAP50"],
            meta["val_mAP50_95"],
            meta["epochs_done"],
            meta["wall_hours"],
        )


if __name__ == "__main__":
    main()
