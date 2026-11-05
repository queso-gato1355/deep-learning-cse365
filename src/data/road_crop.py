"""Build road-region datasets: keep only the bottom ``keep`` fraction of every RDD frame.

Potholes lie on the road, i.e. in the lower part of a vehicle-camera frame, so dropping the top
(sky, buildings) saves inference FLOPs without shrinking potholes. Boxes are clipped to the crop
and dropped if less than ``min_visible`` of their area remains. ``meta.csv`` stores the crop
offset per image so Phase 3 can map predictions back to full-frame coordinates (potholes lost to
the crop then count as misses).

Outputs per keep value (e.g. 0.6 -> ``crop60``):
    data/processed/rdd_crop60/{images,labels,meta.csv}
    data/processed/splits/rdd_crop60_{train,val,test}.txt
    data/processed/rdd_crop60.yaml
"""

from __future__ import annotations

import logging

import cv2
import pandas as pd
from tqdm import tqdm

from src.data.split import write_list, write_yaml
from src.data.synth_rain import read_list
from src.utils.common import base_parser, imread, imwrite, load_config, project_path, setup_logging

log = logging.getLogger("road_crop")


def crop_labels(lines: list[str], keep: float, min_visible: float) -> tuple[list[str], int]:
    """Clip normalized YOLO boxes to the bottom ``keep`` of the image and renormalize.

    Args:
        lines: YOLO label rows of the full frame.
        keep: Kept fraction of the image height (from the bottom).
        min_visible: Minimum fraction of a box's area that must remain inside the crop.

    Returns:
        New label rows (relative to the crop) and the number of dropped boxes.
    """
    top = 1.0 - keep
    out, dropped = [], 0
    for row in lines:
        _, cx, cy, w, h = map(float, row.split())
        y0, y1 = cy - h / 2, cy + h / 2
        ny0, ny1 = max(y0, top), y1
        if h <= 0 or (ny1 - ny0) / h < min_visible:
            dropped += 1
            continue
        ncy, nh = ((ny0 + ny1) / 2 - top) / keep, (ny1 - ny0) / keep
        out.append(f"0 {cx:.6f} {ncy:.6f} {w:.6f} {nh:.6f}")
    return out, dropped


def build(keep: float, min_visible: float, dry_run: bool) -> None:
    """Create the cropped dataset, split lists and dataset YAML for one keep fraction.

    Args:
        keep: Kept fraction of the image height.
        min_visible: Minimum visible area fraction for a box to survive.
        dry_run: Only count, write nothing.
    """
    cfg = load_config("data")
    proc = project_path(cfg["paths"]["processed"])
    splits = project_path(cfg["paths"]["splits"])
    name = f"crop{round(keep * 100)}"
    out = proc / f"rdd_{name}"
    rows, lists = [], {}
    for split in ("train", "val", "test"):
        images = read_list(splits / f"rdd_{split}.txt")
        outs = []
        for img in tqdm(images, desc=f"{name}/{split}"):
            lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
            lines = [r for r in lbl.read_text(encoding="utf-8").splitlines() if r.strip()]
            new, dropped = crop_labels(lines, keep, min_visible)
            dst = out / "images" / img.name
            if not dry_run:
                im = imread(img)
                y0 = int(round(im.shape[0] * (1 - keep)))
                if not dst.exists():
                    imwrite(dst, im[y0:], [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out / "labels").mkdir(parents=True, exist_ok=True)
                (out / "labels" / f"{img.stem}.txt").write_text(
                    "\n".join(new) + ("\n" if new else ""), encoding="utf-8"
                )
            rows.append(
                {
                    "stem": img.stem,
                    "split": split,
                    "keep": keep,
                    "boxes_full": len(lines),
                    "boxes_kept": len(new),
                    "boxes_dropped": dropped,
                }
            )
            outs.append(dst)
        lists[split] = outs
    meta = pd.DataFrame(rows)
    summary = meta.groupby("split")[["boxes_full", "boxes_kept", "boxes_dropped"]].sum()
    summary["kept_%"] = (100 * summary["boxes_kept"] / summary["boxes_full"]).round(1)
    log.info("[%s] boxes kept after cropping:\n%s", name, summary.to_string())
    if dry_run:
        return
    meta.to_csv(out / "meta.csv", index=False)
    for split, outs in lists.items():
        write_list(splits / f"rdd_{name}_{split}.txt", outs, splits)
    write_yaml(
        proc / f"rdd_{name}.yaml",
        proc,
        {s: f"rdd_{name}_{s}.txt" for s in ("train", "val", "test")},
        cfg["classes"]["names"],
    )
    log.info("Wrote %s and %s", out, proc / f"rdd_{name}.yaml")


def main() -> None:
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    parser.add_argument("--keep", type=float, nargs="+", help="Override keep fractions from config")
    args = parser.parse_args()
    setup_logging("road_crop")
    rc = load_config("data")["road_crop"]
    for keep in args.keep or rc["keep"]:
        build(keep, rc["min_visible"], args.dry_run)


if __name__ == "__main__":
    main()
