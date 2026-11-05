"""Convert RDD2022 Pascal-VOC annotations to single-class YOLO labels (D40 -> class 0).

Images are hard-linked (falling back to copying) at their original resolution into
``data/processed/rdd/images``; resizing is left to training (``imgsz``). Negative images (no D40)
are converted too; ``split.py`` subsamples them to ``neg_ratio`` per split.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils.common import base_parser, load_config, project_path, setup_logging

log = logging.getLogger("convert_rdd")

SEQ_RE: re.Pattern[str] = re.compile(r"_(\d+)$")


def find_country_dir(raw_root: Path, country: str) -> Path:
    """Locate ``<country>/train`` (with images and annotations) anywhere under the raw root.

    Args:
        raw_root: Extracted RDD2022 root.
        country: Country folder name.

    Returns:
        The country's ``train`` directory.

    Raises:
        FileNotFoundError: If the folder is not found.
    """
    for cand in sorted(raw_root.rglob(f"{country}/train")):
        if (cand / "images").is_dir() and (cand / "annotations" / "xmls").is_dir():
            return cand
    raise FileNotFoundError(f"{country}/train with images/ and annotations/xmls/ not found")


def parse_voc(xml_path: Path, keep: str) -> tuple[int, int, list[tuple[float, ...]], Counter]:
    """Parse one VOC file.

    Args:
        xml_path: Annotation file.
        keep: Class name to keep (e.g. ``"D40"``).

    Returns:
        ``(width, height, boxes, class_counts)`` where boxes are ``(xmin, ymin, xmax, ymax)``
        for the kept class and class_counts counts every class in the file.
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    boxes: list[tuple[float, ...]] = []
    counts: Counter = Counter()
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        counts[name] += 1
        bb = obj.find("bndbox")
        if name != keep or bb is None:
            continue
        boxes.append(tuple(float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")))
    return w, h, boxes, counts


def to_yolo(boxes: list[tuple[float, ...]], w: int, h: int) -> tuple[list[str], int]:
    """Convert pixel boxes to YOLO lines, clipping to the image and dropping degenerate boxes.

    Args:
        boxes: ``(xmin, ymin, xmax, ymax)`` boxes.
        w: Image width.
        h: Image height.

    Returns:
        YOLO label lines and the number of dropped boxes.
    """
    lines, dropped = [], 0
    for x0, y0, x1, y1 in boxes:
        x0, x1 = max(0.0, min(x0, x1)), min(float(w), max(x0, x1))
        y0, y1 = max(0.0, min(y0, y1)), min(float(h), max(y0, y1))
        if x1 - x0 < 1 or y1 - y0 < 1:
            dropped += 1
            continue
        cx, cy, bw, bh = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, dropped


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst`` (same volume), else copy it.

    Args:
        src: Source file.
        dst: Destination file.
    """
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_country(
    country: str, raw_root: Path, out: Path, keep: str, dry_run: bool
) -> tuple[list[dict], Counter]:
    """Convert one country.

    Args:
        country: Country folder name.
        raw_root: Extracted RDD2022 root.
        out: Output root (``data/processed/rdd``).
        keep: Class to keep.
        dry_run: Parse only.

    Returns:
        Index rows and all-class counts for the country.
    """
    cdir = find_country_dir(raw_root, country)
    images = {p.stem: p for p in (cdir / "images").iterdir() if p.suffix.lower() == ".jpg"}
    rows: list[dict] = []
    counts: Counter = Counter()
    dropped = 0
    for xml in tqdm(sorted((cdir / "annotations" / "xmls").glob("*.xml")), desc=country):
        img = images.get(xml.stem)
        if img is None:
            log.warning("No image for %s", xml.name)
            continue
        w, h, boxes, c = parse_voc(xml, keep)
        counts.update(c)
        if w <= 0 or h <= 0:
            with Image.open(img) as im:
                w, h = im.size
        lines, d = to_yolo(boxes, w, h)
        dropped += d
        m = SEQ_RE.search(xml.stem)
        rows.append(
            {
                "stem": xml.stem,
                "country": country,
                "seq": int(m.group(1)) if m else -1,
                "width": w,
                "height": h,
                "n_boxes": len(lines),
                "negative": len(lines) == 0,
            }
        )
        if dry_run:
            continue
        link_or_copy(img, out / "images" / img.name)
        (out / "labels" / f"{xml.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    if dropped:
        log.warning("[%s] dropped %d degenerate %s boxes", country, dropped, keep)
    return rows, counts


def main() -> None:
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    args = parser.parse_args()
    setup_logging("convert_rdd")
    cfg = load_config("data")
    rdd = cfg["rdd"]
    raw_root = project_path(cfg["paths"]["raw_rdd"])
    out = project_path(cfg["paths"]["processed_rdd"])
    for name, reason in rdd["excluded_countries"].items():
        log.info("Excluding %s: %s", name, reason)
    if not args.dry_run:
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for country in rdd["countries"]:
        r, counts = convert_country(country, raw_root, out, rdd["source_class"], args.dry_run)
        rows += r
        log.info("[%s] %d annotated images; classes %s", country, len(r), dict(counts))
    index = pd.DataFrame(rows)
    summary = index.groupby("country").agg(
        images=("stem", "size"),
        positives=("negative", lambda s: int((~s).sum())),
        d40_boxes=("n_boxes", "sum"),
    )
    log.info("RDD2022 %s summary:\n%s", rdd["source_class"], summary.to_string())
    if not args.dry_run:
        index.to_csv(out / "index.csv", index=False)
        log.info("Wrote %s", out / "index.csv")


if __name__ == "__main__":
    main()
