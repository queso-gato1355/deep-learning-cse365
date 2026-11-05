"""Convert MWPD to single-class YOLO data and write ``data/processed/mwpd/meta.csv``.

Weather/lighting come from ``data/processed/mwpd/weather_tags.csv`` (manual tags, because MWPD
encodes neither in folders, file names nor metadata; see ``inspect.py``). Files are renamed
``{orig_split}__{file}`` so names stay unique after the original splits are merged.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.data.convert_rdd import link_or_copy
from src.data.inspect import IMG_EXTS
from src.utils.common import base_parser, load_config, project_path, setup_logging

log = logging.getLogger("convert_mwpd")


def convert_label(src: Path) -> tuple[list[str], int]:
    """Map every box to class 0 and clip coordinates to [0, 1].

    Args:
        src: YOLO label file.

    Returns:
        Converted lines and the number of malformed rows dropped.
    """
    lines, bad = [], 0
    for row in src.read_text(encoding="utf-8").splitlines():
        parts = row.split()
        if not parts:
            continue
        if len(parts) != 5:
            bad += 1
            continue
        cx, cy, w, h = (min(max(float(v), 0.0), 1.0) for v in parts[1:])
        if w <= 0 or h <= 0:
            bad += 1
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines, bad


def main() -> None:
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    args = parser.parse_args()
    setup_logging("convert_mwpd")
    cfg = load_config("data")
    raw = project_path(cfg["paths"]["raw_mwpd"])
    out = project_path(cfg["paths"]["processed_mwpd"])
    tags_path = out / "weather_tags.csv"
    if not tags_path.exists():
        log.error("Missing %s. Run `python -m src.data.tag_weather` first.", tags_path)
        sys.exit(2)
    tags = pd.read_csv(tags_path)
    if not tags["weather"].isin(["normal", "rainy"]).all():
        log.error(
            "weather_tags.csv has untagged/invalid rows: %d",
            (~tags["weather"].isin(["normal", "rainy"])).sum(),
        )
        sys.exit(2)
    tag_map = {(r.split, r.file): r for r in tags.itertuples()}

    rows, bad_total = [], 0
    for img in sorted(p for p in raw.rglob("images/*") if p.suffix.lower() in IMG_EXTS):
        split = img.parent.parent.name
        tag = tag_map.get((split, img.name))
        if tag is None:
            log.error("No weather tag for %s/%s", split, img.name)
            sys.exit(2)
        lines, bad = convert_label(img.parent.parent / "labels" / f"{img.stem}.txt")
        bad_total += bad
        name = f"{split}__{img.name}"
        rows.append(
            {
                "stem": Path(name).stem,
                "orig_split": split,
                "file": img.name,
                "weather": tag.weather,
                "lighting": tag.lighting,
                "tagged_by": getattr(tag, "tagged_by", "unknown"),
                "n_boxes": len(lines),
            }
        )
        if args.dry_run:
            continue
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        link_or_copy(img, out / "images" / name)
        (out / "labels" / f"{Path(name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    meta = pd.DataFrame(rows)
    if bad_total:
        log.warning("Dropped %d malformed label rows", bad_total)
    log.info(
        "MWPD weather x lighting:\n%s",
        pd.crosstab(meta["weather"], meta["lighting"], margins=True).to_string(),
    )
    log.info("Tag source: %s", meta["tagged_by"].value_counts().to_dict())
    if not args.dry_run:
        meta.to_csv(out / "meta.csv", index=False)
        log.info("Wrote %s", out / "meta.csv")


if __name__ == "__main__":
    main()
