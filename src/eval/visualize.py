"""Phase 3 example figures: how each model draws its boxes, and what each layer looks at.

Two kinds of figures, every tile being the 640x640 square letterboxed network input:

``fig_pred_grid``: rows = sample RDD val images (one per country), columns = ground truth + one
    run per configuration (lowest seed). Boxes come from the cached val predictions of
    ``evaluate.py`` at the run's val-fixed confidence threshold: green = correct (TP), red = false
    alarm (FP), yellow = missed pothole (FN). Crop models get the region they never see shaded.

``fig_layers_{config}_{image}``: one model, one image, layer by layer:
    row 1  input + backbone feature maps at strides 4/8/16/32 (mean |activation|),
    row 2  neck outputs fed to each detection head (P2/P3/P4/P5),
    row 3  each head's pothole-score map with the boxes that head alone proposes, then the final
           prediction. Shows which scale finds which pothole size. (YOLO models only.)

Outputs: ``outputs/figures/fig_pred_grid.{png,pdf}``, ``outputs/figures/layers/*.png``.

Usage:
    python -m src.eval.visualize                 # both figure types
    python -m src.eval.visualize --only grid     # or: --only layers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import patches

from src.data.synth_rain import read_list
from src.eval.evaluate import crop_keep, label_path, read_gt, score
from src.eval.speed import configurations
from src.utils.common import base_parser, imread, load_config, project_path, setup_logging

log = logging.getLogger("visualize")
SIZE = 640
GREEN, RED, YELLOW = "#2ecc40", "#ff4136", "#ffdc00"


def letterbox(im: np.ndarray, size: int = SIZE) -> tuple[np.ndarray, float, int, int]:
    """Resize the longest side to ``size`` and pad to a centred square (Ultralytics grey 114).

    Args:
        im: BGR image.
        size: Square side.

    Returns:
        Square image, scale, x padding, y padding.
    """
    h, w = im.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    px, py = (size - nw) // 2, (size - nh) // 2
    sq = np.full((size, size, 3), 114, np.uint8)
    sq[py : py + nh, px : px + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return sq, r, px, py


def to_square(boxes: np.ndarray, y0: int, r: float, px: int, py: int) -> np.ndarray:
    """Map full-frame ``xyxy`` boxes into the square tile of a (possibly cropped) image.

    Args:
        boxes: ``(n, 4)`` full-frame boxes.
        y0: Rows removed from the top before letterboxing (0 for full frames).
        r: Letterbox scale.
        px: x padding.
        py: y padding.

    Returns:
        ``(n, 4)`` boxes in tile pixels.
    """
    b = boxes.astype(np.float32).copy().reshape(-1, 4)
    b[:, [1, 3]] -= y0
    return b * r + np.array([px, py, px, py], np.float32)


def classify(pred: np.ndarray, conf: np.ndarray, gt: np.ndarray, thr: float) -> tuple:
    """Greedy IoU matching (highest confidence first) for TP/FP/FN colouring.

    Args:
        pred: ``(n, 4)`` predicted boxes.
        conf: ``(n,)`` confidences.
        gt: ``(m, 4)`` ground-truth boxes.
        thr: IoU threshold.

    Returns:
        ``(tp_mask over pred, fn_mask over gt)``.
    """
    from src.eval.evaluate import box_iou

    tp = np.zeros(len(pred), bool)
    hit = np.zeros(len(gt), bool)
    if len(pred) and len(gt):
        iou = box_iou(pred, gt)
        for i in np.argsort(-conf):
            cand = np.where(hit, 0, iou[i])
            j = int(cand.argmax())
            if cand[j] >= thr:
                tp[i], hit[j] = True, True
    return tp, ~hit


def draw_boxes(ax: plt.Axes, boxes: np.ndarray, color: str, lw: float = 1.4, ls: str = "-") -> None:
    """Draw ``xyxy`` rectangles.

    Args:
        ax: Axes.
        boxes: ``(n, 4)`` tile-pixel boxes.
        color: Edge colour.
        lw: Line width.
        ls: Line style.
    """
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(
            patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=lw, ls=ls)
        )


def tile(ax: plt.Axes, sq_bgr: np.ndarray, title: str, fontsize: int = 7) -> None:
    """Show a square BGR image without axes.

    Args:
        ax: Axes.
        sq_bgr: Square image.
        title: Title.
        fontsize: Title size.
    """
    ax.imshow(cv2.cvtColor(sq_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)
    ax.axis("off")


def sample_images(ecfg: dict) -> list[tuple[int, Path]]:
    """Pick sample val images: given stems, else one random labelled image per country.

    Args:
        ecfg: Eval config.

    Returns:
        ``(index in the val list, path)`` pairs.
    """
    vcfg = ecfg["visualize"]
    val = read_list(project_path(ecfg["val_list"]))
    if vcfg.get("stems"):
        idx = {p.stem: i for i, p in enumerate(val)}
        return [(idx[s], val[idx[s]]) for s in vcfg["stems"]]
    rng = np.random.default_rng(vcfg["seed"])
    by_country: dict[str, list[int]] = {}
    for i, p in enumerate(val):
        if label_path(p).read_text(encoding="utf-8").strip():
            by_country.setdefault(p.stem.rsplit("_", 1)[0], []).append(i)
    picks = [int(rng.choice(v)) for _, v in sorted(by_country.items())]
    return [(i, val[i]) for i in picks[: vcfg["n_images"]]]


def run_preds(run_id: str, ecfg: dict) -> tuple[dict[str, np.ndarray], float]:
    """Cached val predictions of a run and its val-fixed confidence threshold.

    Args:
        run_id: Run id.
        ecfg: Eval config.

    Returns:
        Prediction arrays and threshold.

    Raises:
        FileNotFoundError: If ``evaluate.py`` has not been run for this run.
    """
    f = project_path(ecfg["paths"]["preds"]) / run_id / "val.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run `python -m src.eval.evaluate` first")
    p = dict(np.load(f))
    return p, score(p, None)["conf_thr"]


def fig_pred_grid(ecfg: dict, configs: list[str] | None = None) -> plt.Figure:
    """Grid of sample images (rows) x models (columns) with TP/FP/FN-coloured boxes.

    Args:
        ecfg: Eval config.
        configs: Configurations to show (default: every evaluated one, ladder order).

    Returns:
        Figure.
    """
    from src.eval.aggregate import ORDER

    metas = {m["config"]: m for m in configurations(project_path("outputs/runs"))}
    avail = [
        c for c in metas if (project_path(ecfg["paths"]["preds"]) / metas[c]["run_id"]).exists()
    ]
    configs = configs or sorted(avail, key=lambda c: ORDER.index(c) if c in ORDER else 99)
    samples = sample_images(ecfg)
    preds = {c: run_preds(metas[c]["run_id"], ecfg) for c in configs}
    iou_thr = ecfg["visualize"]["iou_match"]
    nr, nc = len(samples), len(configs) + 1
    fig, axes = plt.subplots(nr, nc, figsize=(1.75 * nc, 1.85 * nr), squeeze=False)
    for r_i, (idx, path) in enumerate(samples):
        im = imread(path)
        h, w = im.shape[:2]
        gt = read_gt(path, h, w)
        sq, r, px, py = letterbox(im)
        ax = axes[r_i, 0]
        tile(ax, sq, f"GT: {path.stem}\n{len(gt)} pothole(s)", 6)
        draw_boxes(ax, to_square(gt, 0, r, px, py), GREEN)
        for c_i, c in enumerate(configs, start=1):
            p, thr = preds[c]
            sel = (p["pred_img"] == idx) & (p["pred_conf"] >= thr)
            pb, pc = p["pred_box"][sel], p["pred_conf"][sel]
            tp, fn = classify(pb, pc, gt, iou_thr)
            ax = axes[r_i, c_i]
            tile(
                ax,
                sq,
                f"{c}\nTP {tp.sum()} FP {(~tp).sum()} FN {fn.sum()}"
                if r_i == 0
                else f"TP {tp.sum()} FP {(~tp).sum()} FN {fn.sum()}",
                6,
            )
            keep = crop_keep(metas[c]["data_key"])
            if keep:  # region above the crop line is never seen by this model
                y_line = py + r * round(h * (1 - keep))
                ax.add_patch(patches.Rectangle((0, 0), SIZE, y_line, fc="black", alpha=0.45))
                ax.axhline(y_line, color="white", lw=0.8, ls="--")
            draw_boxes(ax, to_square(gt[fn], 0, r, px, py), YELLOW, 1.2, "--")
            draw_boxes(ax, to_square(pb[~tp], 0, r, px, py), RED)
            draw_boxes(ax, to_square(pb[tp], 0, r, px, py), GREEN)
    fig.suptitle(
        "Predictions on RDD val images at each run's val-fixed confidence threshold "
        "(green = TP, red = FP, yellow dashed = missed; shaded = outside the crop)",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def heat(sq: np.ndarray, fmap: np.ndarray, absolute: bool = False) -> np.ndarray:
    """Overlay a 2-D map (resized to the tile) on the square image.

    Args:
        sq: Square BGR image.
        fmap: ``(H, W)`` map.
        absolute: Treat ``fmap`` as already in [0, 1] (scores) instead of min-max scaling.

    Returns:
        BGR overlay.
    """
    m = fmap.astype(np.float32)
    if not absolute:
        lo, hi = np.percentile(m, 1), np.percentile(m, 99)
        m = (m - lo) / max(hi - lo, 1e-6)
    m = np.clip(m, 0, 1)
    m = cv2.resize(
        m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST if absolute else cv2.INTER_LINEAR
    )
    col = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if absolute:  # scores: transparent where ~0 so the photo stays visible
        a = (0.85 * np.sqrt(m))[..., None]
        return (sq * (1 - a) + col * a).astype(np.uint8)
    return cv2.addWeighted(sq, 0.45, col, 0.55, 0)


def layer_maps(model: torch.nn.Module, sq: np.ndarray) -> tuple[dict[int, torch.Tensor], Any]:
    """Forward the square image and capture every top-level layer's output.

    Args:
        model: Ultralytics ``DetectionModel`` (eval, fp32).
        sq: Square BGR image.

    Returns:
        ``{layer index: output}`` and the raw model output.
    """
    dev = next(model.parameters()).device
    x = torch.from_numpy(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None]
    x = x.float().div(255).to(dev)
    outs: dict[int, torch.Tensor] = {}
    hooks = [
        layer.register_forward_hook(lambda _m, _i, o, k=k: outs.__setitem__(k, o))
        for k, layer in enumerate(model.model)
    ]
    with torch.no_grad():
        y = model(x)
    for hk in hooks:
        hk.remove()
    return outs, y


def fig_layers(meta: dict, idx: int, path: Path, ecfg: dict) -> plt.Figure:
    """Layer-by-layer view of one YOLO run on one image.

    Args:
        meta: Run metadata (from :func:`configurations`).
        idx: Index of the image in the val list (for the cached final prediction).
        path: Image path.
        ecfg: Eval config.

    Returns:
        Figure.
    """
    from torchvision.ops import nms
    from ultralytics import YOLO

    model = YOLO(str(project_path("outputs/runs") / meta["run_id"] / "weights/best.pt")).model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.float().eval().to(dev)
    im = imread(path)
    h, w = im.shape[:2]
    keep = crop_keep(meta["data_key"])
    y0 = int(round(h * (1 - keep))) if keep else 0
    sq, r, px, py = letterbox(im[y0:])
    gt = read_gt(path, h, w)
    gt_sq = to_square(gt, y0, r, px, py)
    outs, y = layer_maps(model, sq)

    head = model.model[-1]
    first_up = next(i for i, m in enumerate(model.model) if type(m).__name__ == "Upsample")
    last_at: dict[int, int] = {}  # backbone: last layer at each stride
    for i in range(first_up):
        last_at[SIZE // outs[i].shape[-1]] = i
    stages = [(s, last_at[s]) for s in sorted(last_at) if s >= 4]
    levels = [(int(s), f) for s, f in zip(head.stride.tolist(), head.f, strict=True)]

    raw = (y[0] if isinstance(y, (tuple, list)) else y)[0]  # (4 + nc, N): xywh + scores
    sizes = [outs[f].shape[-2] * outs[f].shape[-1] for _, f in levels]
    p, thr = run_preds(meta["run_id"], ecfg)

    ncol = max(len(stages) + 1, len(levels) + 1)
    fig, axes = plt.subplots(3, ncol, figsize=(2.3 * ncol, 2.55 * 3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    ax = axes[0, 0]
    tile(ax, sq, f"input {SIZE}x{SIZE}" + (f" (bottom {keep:.0%})" if keep else "") + "\nGT green")
    draw_boxes(ax, gt_sq, GREEN)
    for c, (s, li) in enumerate(stages, start=1):
        f = outs[li][0]
        tile(
            axes[0, c],
            heat(sq, f.abs().mean(0).cpu().numpy()),
            f"backbone L{li} {type(model.model[li]).__name__}\nstride {s}, {f.shape[0]} ch",
        )
    for c, (s, li) in enumerate(levels):
        f = outs[li][0]
        tile(
            axes[1, c],
            heat(sq, f.abs().mean(0).cpu().numpy()),
            f"neck L{li} -> head P{int(np.log2(s))}\nstride {s}, {f.shape[1]}x{f.shape[2]}",
        )
    start = 0
    for c, ((s, li), n) in enumerate(zip(levels, sizes, strict=True)):
        hh, ww = outs[li].shape[-2:]
        seg = raw[:, start : start + n]
        start += n
        sc = seg[4:].max(0).values
        tile(axes[2, c], heat(sq, sc.reshape(hh, ww).cpu().numpy(), absolute=True), "")
        m = sc >= thr
        xywh = seg[:4, m].T
        b = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
        b = b[nms(b, sc[m], 0.7)].cpu().numpy() if len(b) else np.zeros((0, 4))
        draw_boxes(axes[2, c], gt_sq, GREEN, 0.8, "--")
        draw_boxes(axes[2, c], b, "white", 1.4)
        axes[2, c].set_title(
            f"head P{int(np.log2(s))} score map (0-1)\n{len(b)} box(es) >= conf {thr:.2f}",
            fontsize=7,
        )
    sel = (p["pred_img"] == idx) & (p["pred_conf"] >= thr)
    pb, pc = p["pred_box"][sel], p["pred_conf"][sel]
    tp, fn = classify(pb, pc, gt, ecfg["visualize"]["iou_match"])
    ax = axes[2, len(levels)]
    tile(ax, sq, f"final prediction\nTP {tp.sum()} FP {(~tp).sum()} FN {fn.sum()}")
    draw_boxes(ax, to_square(gt[fn], y0, r, px, py), YELLOW, 1.2, "--")
    draw_boxes(ax, to_square(pb[~tp], y0, r, px, py), RED)
    draw_boxes(ax, to_square(pb[tp], y0, r, px, py), GREEN)
    for row, label in enumerate(["backbone", "neck", "detection heads"]):
        fig.text(0.004, 1 - (row + 0.5) / 3, label, rotation=90, va="center", fontsize=9)
    fig.suptitle(f"{meta['config']} on {path.stem}: layer by layer", fontsize=10)
    fig.tight_layout(rect=(0.015, 0, 1, 1))
    del model
    return fig


def main() -> None:
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    parser.add_argument("--only", choices=["grid", "layers"], help="Only one figure type")
    args = parser.parse_args()
    setup_logging("visualize")
    ecfg = load_config("eval")
    figs = project_path(ecfg["paths"]["figures"])
    samples = sample_images(ecfg)
    log.info("Sample val images: %s", ", ".join(p.stem for _, p in samples))
    if args.dry_run:
        return
    if args.only in (None, "grid"):
        fig = fig_pred_grid(ecfg)
        for ext in ("png", "pdf"):
            fig.savefig(figs / f"fig_pred_grid.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", figs / "fig_pred_grid.png")
    if args.only in (None, "layers"):
        metas = {m["config"]: m for m in configurations(project_path("outputs/runs"))}
        out = figs / "layers"
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for c in ecfg["visualize"]["layer_configs"]:
            if c not in metas:
                log.warning("[%s] not trained yet - skipped", c)
                continue
            for idx, path in samples[: ecfg["visualize"]["layer_images"]]:
                fig = fig_layers(metas[c], idx, path, ecfg)
                f = out / f"fig_layers_{c}_{path.stem}.png"
                fig.savefig(f, dpi=150, bbox_inches="tight")
                plt.close(fig)
                rows.append({"config": c, "image": path.stem, "file": f.name})
                log.info("Wrote %s", f)
        pd.DataFrame(rows).to_csv(out / "index.csv", index=False)


if __name__ == "__main__":
    main()
