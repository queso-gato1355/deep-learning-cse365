"""Inspect raw datasets: folder tree, file counts, label format, missing pairs, MWPD conditions.

Run as a module (``python -m src.data.inspect``) so this file never shadows the stdlib ``inspect``.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from src.utils.common import base_parser, load_config, project_path, setup_logging

log = logging.getLogger("inspect")

IMG_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
LABEL_EXTS: frozenset[str] = frozenset({".xml", ".txt", ".json"})
WEATHER_PATTERNS: dict[str, re.Pattern[str]] = {
    "rainy": re.compile(r"(?<![a-z])(rain|rainy|wet|storm|monsoon)(?![a-z])", re.I),
    "normal": re.compile(r"(?<![a-z])(normal|clear|sunny|dry)(?![a-z])", re.I),
}
LIGHTING_PATTERNS: dict[str, re.Pattern[str]] = {
    "night": re.compile(r"night", re.I),
    "twilight": re.compile(r"twilight|dusk|dawn|evening", re.I),
    "day": re.compile(r"(?<![a-z])day", re.I),
}
ROBOFLOW_SUFFIX: re.Pattern[str] = re.compile(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", re.I)


def folder_tree(root: Path, max_depth: int = 4) -> list[str]:
    """Render a directory tree with per-folder file counts.

    Args:
        root: Directory to walk.
        max_depth: Maximum depth to print.

    Returns:
        Lines of the tree.
    """
    lines: list[str] = []
    for d in sorted([root, *[p for p in root.rglob("*") if p.is_dir()]]):
        depth = len(d.relative_to(root).parts)
        if depth > max_depth:
            continue
        n = sum(1 for p in d.iterdir() if p.is_file())
        lines.append(f"{'  ' * depth}{d.name}/ ({n} files)")
    return lines


def detect_label_format(root: Path) -> str:
    """Guess the annotation format from the label files present.

    Args:
        root: Dataset root.

    Returns:
        ``"voc_xml"``, ``"yolo_txt"``, ``"coco_json"``, a ``+``-joined mix, or ``"none"``.
    """
    found: set[str] = set()
    if any(root.rglob("*.xml")):
        found.add("voc_xml")
    for txt in root.rglob("*.txt"):
        rows = [r.split() for r in txt.read_text(encoding="utf-8").splitlines() if r.strip()]
        if rows and all(len(r) == 5 and all(_is_number(x) for x in r) for r in rows):
            found.add("yolo_txt")
            break
    for js in root.rglob("*.json"):
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and {"images", "annotations"} <= data.keys():
            found.add("coco_json")
    return "+".join(sorted(found)) or "none"


def _is_number(s: str) -> bool:
    """Return True if ``s`` parses as a float."""
    try:
        float(s)
    except ValueError:
        return False
    return True


def find_label_dir(images_dir: Path) -> Path | None:
    """Locate the label folder paired with an ``images`` folder.

    Args:
        images_dir: A directory named ``images``.

    Returns:
        Sibling ``labels`` or ``annotations/xmls`` directory, or None.
    """
    for cand in (images_dir.parent / "labels", images_dir.parent / "annotations" / "xmls"):
        if cand.is_dir():
            return cand
    return None


def missing_pairs(root: Path) -> pd.DataFrame:
    """List images without labels and labels without images.

    Folders named ``images`` without any label folder (e.g. unlabeled official test sets)
    are reported once as a whole rather than file by file.

    Args:
        root: Dataset root.

    Returns:
        DataFrame with columns ``kind, split_dir, file``.
    """
    rows: list[dict[str, str]] = []
    for images_dir in sorted(p for p in root.rglob("images") if p.is_dir()):
        label_dir = find_label_dir(images_dir)
        rel = images_dir.parent.relative_to(root).as_posix()
        if label_dir is None:
            rows.append({"kind": "unlabeled_folder", "split_dir": rel, "file": "*"})
            continue
        imgs = {p.stem: p.name for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        labels = {p.stem: p.name for p in label_dir.iterdir() if p.suffix.lower() in LABEL_EXTS}
        rows += [
            {"kind": "image_without_label", "split_dir": rel, "file": imgs[s]}
            for s in sorted(imgs.keys() - labels.keys())
        ]
        rows += [
            {"kind": "label_without_image", "split_dir": rel, "file": labels[s]}
            for s in sorted(labels.keys() - imgs.keys())
        ]
    return pd.DataFrame(rows, columns=["kind", "split_dir", "file"])


def match_condition(text: str, patterns: dict[str, re.Pattern[str]]) -> str:
    """Return the first condition whose pattern matches ``text``, else ``"unknown"``.

    Args:
        text: Path string to search.
        patterns: Condition name -> regex.

    Returns:
        Condition name.
    """
    for name, pat in patterns.items():
        if pat.search(text):
            return name
    return "unknown"


def source_stem(name: str) -> str:
    """Strip a Roboflow export suffix (``_jpg.rf.<hash>``) to recover the original frame name.

    Args:
        name: File stem.

    Returns:
        Original source stem.
    """
    return ROBOFLOW_SUFFIX.sub("", name)


def inspect_mwpd_conditions(root: Path) -> pd.DataFrame:
    """Try to extract weather and lighting tags from MWPD paths.

    Args:
        root: MWPD raw root.

    Returns:
        One row per image with ``split, file, source, weather, lighting``.
    """
    rows = []
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS):
        rel = img.relative_to(root).as_posix()
        rows.append(
            {
                "split": img.parent.parent.name,
                "file": img.name,
                "source": source_stem(img.stem),
                "weather": match_condition(rel, WEATHER_PATTERNS),
                "lighting": match_condition(rel, LIGHTING_PATTERNS),
            }
        )
    return pd.DataFrame(rows)


def inspect_dataset(name: str, root: Path, tables: Path) -> None:
    """Log tree, extension counts and label format; save the missing-pair table.

    Args:
        name: Dataset key (``rdd2022`` or ``mwpd``).
        root: Raw dataset root.
        tables: Output table directory.
    """
    if not root.exists() or not any(root.iterdir()):
        log.warning("[%s] %s is missing or empty - skipped", name, root)
        return
    log.info("[%s] folder tree:\n%s", name, "\n".join(folder_tree(root)))
    ext = Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file())
    log.info("[%s] files by extension: %s", name, dict(ext.most_common()))
    log.info("[%s] label format: %s", name, detect_label_format(root))
    miss = missing_pairs(root)
    out = tables / f"missing_{name}.csv"
    miss.to_csv(out, index=False)
    log.info("[%s] missing pairs: %s -> %s", name, miss["kind"].value_counts().to_dict(), out)


def main() -> None:
    """CLI entry point. Exits with code 2 if MWPD weather cannot be derived and no tags exist."""
    parser = base_parser(__doc__ or "")
    parser.parse_args()
    setup_logging("inspect")
    cfg = load_config("data")
    tables = project_path(cfg["paths"]["tables"])
    tables.mkdir(parents=True, exist_ok=True)

    rdd_root = project_path(cfg["paths"]["raw_rdd"])
    mwpd_root = project_path(cfg["paths"]["raw_mwpd"])
    inspect_dataset("rdd2022", rdd_root, tables)
    inspect_dataset("mwpd", mwpd_root, tables)

    if not mwpd_root.exists():
        return
    cond = inspect_mwpd_conditions(mwpd_root)
    log.info("[mwpd] images per split: %s", cond["split"].value_counts().to_dict())
    log.info(
        "[mwpd] Roboflow copies: %d images from %d unique source frames",
        len(cond),
        cond["source"].nunique(),
    )
    log.info("[mwpd] weather from paths: %s", cond["weather"].value_counts().to_dict())
    log.info("[mwpd] lighting from paths: %s", cond["lighting"].value_counts().to_dict())
    log.info(
        "[mwpd] split x weather:\n%s",
        pd.crosstab(cond["split"], cond["weather"], margins=True).to_string(),
    )
    cond.to_csv(tables / "mwpd_path_conditions.csv", index=False)

    tags = project_path(cfg["paths"]["processed_mwpd"]) / "weather_tags.csv"
    if (cond["weather"] == "unknown").any():
        if tags.exists():
            log.info("[mwpd] weather not in paths; using manual tags from %s", tags)
            return
        log.error(
            "[mwpd] Weather cannot be derived from folders, file names or metadata "
            "(%d/%d unknown). Run `python -m src.data.tag_weather`, tag images in "
            "outputs/tag_weather.html and save the exported CSV to %s.",
            (cond["weather"] == "unknown").sum(),
            len(cond),
            tags,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
