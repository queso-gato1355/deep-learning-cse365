"""Phase 3 speed benchmark: batch-1 latency per model configuration on GPU (FP32/FP16) and CPU.

One configuration = (model, dataset, imgsz); seeds share the architecture, so the lowest-seed run's
``best.pt`` is timed (KD runs are a yolo26n and time like one). Images come from test set A;
road-crop configurations get the cropped frame, so their smaller input shows up as real time saved.

Ultralytics ``predict`` reports per-image preprocess / inference / postprocess (NMS for YOLOv8,
none for the NMS-free YOLO26 head) times with CUDA synchronisation. ``GFLOPs_eff`` is the mean
forward cost at the actual letterboxed input shapes (rectangular, multiple of 32), not at 640x640.

Writes ``outputs/results/speed.csv``.

Usage:
    python -m src.eval.speed                     # every configuration, every device
    python -m src.eval.speed --devices gpu       # GPU only
    python -m src.eval.speed --quick             # 20 timed images, 5 warm-up (pipeline check)
"""

from __future__ import annotations

import json
import logging
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.synth_rain import read_list
from src.eval.evaluate import crop_keep
from src.utils.common import base_parser, imread, load_config, project_path, setup_logging
from src.utils.flops import count_gflops

log = logging.getLogger("speed")


def configurations(runs_dir: Path) -> list[dict[str, Any]]:
    """One run per (model, data, imgsz, teacher) configuration: the lowest seed.

    Args:
        runs_dir: ``outputs/runs``.

    Returns:
        Run metadata dictionaries with an added ``config`` key.
    """
    best: dict[str, dict] = {}
    for f in sorted(runs_dir.glob("*/run_meta.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if not (f.parent / "weights" / "best.pt").exists():
            continue
        m["config"] = m["run_id"].rsplit("_s", 1)[0]
        if m["config"] not in best or m["seed"] < best[m["config"]]["seed"]:
            best[m["config"]] = m
    return list(best.values())


def letterbox_hw(h: int, w: int, imgsz: int, stride: int = 32) -> tuple[int, int]:
    """Network input shape for a ``h x w`` image under Ultralytics rectangular letterboxing.

    Args:
        h: Image height.
        w: Image width.
        imgsz: Longest side after resizing.
        stride: Padding multiple.

    Returns:
        ``(height, width)`` of the network input.
    """
    r = imgsz / max(h, w)
    nh, nw = round(h * r), round(w * r)
    return math.ceil(nh / stride) * stride, math.ceil(nw / stride) * stride


def hardware() -> dict[str, str]:
    """Describe the GPU and CPU used for timing.

    Returns:
        ``gpu`` and ``cpu`` names plus CPU thread count.
    """
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    cpu = platform.processor()
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        cpu = out or cpu
    except (OSError, subprocess.SubprocessError):
        pass
    return {"gpu": gpu, "cpu": cpu, "cpu_threads": str(torch.get_num_threads())}


def time_config(
    meta: dict, images: list[np.ndarray], device: str, prec: str, n: int, warmup: int, conf: float
) -> dict[str, float]:
    """Time batch-1 predictions of one configuration.

    Args:
        meta: Run metadata.
        images: Pre-loaded (and, for crop runs, already cropped) BGR images.
        device: ``gpu`` or ``cpu``.
        prec: ``fp32`` or ``fp16``.
        n: Timed predictions.
        warmup: Untimed predictions first.
        conf: Confidence threshold.

    Returns:
        Mean milliseconds per stage, total, FPS and wall-clock FPS.
    """
    from ultralytics import RTDETR, YOLO

    cls = RTDETR if "rtdetr" in meta["cfg"] else YOLO
    model = cls(str(project_path("outputs/runs") / meta["run_id"] / "weights/best.pt"))
    kw = {
        "imgsz": meta["imgsz"],
        "conf": conf,
        "device": 0 if device == "gpu" else "cpu",
        "quantize": 16 if prec == "fp16" else None,
        "verbose": False,
    }
    for i in range(warmup):
        model.predict(images[i % len(images)], **kw)
    speeds = []
    if device == "gpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        speeds.append(model.predict(images[i % len(images)], **kw)[0].speed)
    if device == "gpu":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    s = pd.DataFrame(speeds).mean()
    total = s["preprocess"] + s["inference"] + s["postprocess"]
    del model
    if device == "gpu":
        torch.cuda.empty_cache()
    return {
        "pre_ms": s["preprocess"],
        "inf_ms": s["inference"],
        "post_ms": s["postprocess"],
        "total_ms": total,
        "fps": 1000 / total,
        "wall_fps": n / wall,
    }


def effective_gflops(meta: dict, images: list[np.ndarray]) -> float:
    """Mean forward GFLOPs over the letterboxed shapes of ``images``.

    Args:
        meta: Run metadata.
        images: Images as fed to the model.

    Returns:
        GFLOPs per image.
    """
    from ultralytics import RTDETR, YOLO

    cls = RTDETR if "rtdetr" in meta["cfg"] else YOLO
    weights = project_path("outputs/runs") / meta["run_id"] / "weights/best.pt"
    model = cls(str(weights)).model.float().eval()
    shapes = [letterbox_hw(*im.shape[:2], meta["imgsz"]) for im in images]
    cost = {hw: count_gflops(model, hw) for hw in set(shapes)}
    return float(np.mean([cost[hw] for hw in shapes]))


def main() -> None:
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    parser.add_argument("--devices", nargs="+", choices=["gpu", "cpu"], help="Only these devices")
    parser.add_argument("--configs", nargs="+", help="Only these configurations (run id w/o seed)")
    args = parser.parse_args()
    setup_logging("speed")
    ecfg = load_config("eval")
    scfg = ecfg["speed"]
    metas = configurations(project_path("outputs/runs"))
    if args.configs:
        metas = [m for m in metas if m["config"] in args.configs]
    pairs = [tuple(p) for p in scfg["devices"]]
    if args.devices:
        pairs = [p for p in pairs if p[0] in args.devices]
    if not torch.cuda.is_available():
        log.warning("CUDA not available: GPU timings skipped")
        pairs = [p for p in pairs if p[0] != "gpu"]
    hw = hardware()
    log.info("%d configurations x %s on %s / %s", len(metas), pairs, hw["gpu"], hw["cpu"])
    if args.dry_run:
        return
    n_load = max(scfg["images"], scfg["cpu_images"])
    paths = read_list(project_path(ecfg["testsets"][scfg["testset"]]))[:n_load]
    full = [imread(p) for p in paths]
    rows = []
    for m in metas:
        keep = crop_keep(m["data_key"])
        ims = [im[int(round(im.shape[0] * (1 - keep))) :] for im in full] if keep else full
        gfl = effective_gflops(m, ims)
        for device, prec in pairs:
            gpu = device == "gpu"
            n = scfg["images"] if gpu else scfg["cpu_images"]
            w = scfg["warmup"] if gpu else scfg["cpu_warmup"]
            if args.quick:
                n, w = 20, 5
            t = time_config(m, ims, device, prec, n, w, scfg["conf"])
            rows.append(
                {
                    "config": m["config"],
                    "exp": m["exp"],
                    "model": m["model"],
                    "data_key": m["data_key"],
                    "imgsz": m["imgsz"],
                    "device": device,
                    "precision": prec,
                    "params_M": m.get("params_M"),
                    "gflops_eff": round(gfl, 3),
                    "n_timed": n,
                    **{k: round(v, 3) for k, v in t.items()},
                    "hardware": hw["gpu"] if gpu else f"{hw['cpu']} ({hw['cpu_threads']} threads)",
                }
            )
            log.info(
                "[%s] %s %s: %.2f ms/img (pre %.2f, inf %.2f, post %.2f), %.0f FPS, %.2f GFLOPs",
                m["config"],
                device,
                prec,
                t["total_ms"],
                t["pre_ms"],
                t["inf_ms"],
                t["post_ms"],
                t["fps"],
                gfl,
            )
    out = project_path(ecfg["paths"]["speed"])
    if args.quick:
        out = out.with_name(out.stem + "_quick.csv")
    new = pd.DataFrame(rows)
    if out.exists() and not new.empty:  # keep configurations/devices not timed this time
        old = pd.read_csv(out)
        key = ["config", "device", "precision"]
        old = old.merge(new[key], on=key, how="left", indicator=True)
        new = pd.concat([old[old["_merge"] == "left_only"].drop(columns="_merge"), new])
    out.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(["config", "device", "precision"]).to_csv(out, index=False)
    log.info("Wrote %d rows to %s", len(new), out)


if __name__ == "__main__":
    main()
