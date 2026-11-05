"""Generate synthetic-rain test sets B1-B3 from RDD test (A), once, with fixed seeds.

Only the RDD *test* split is used here; training-time rain (strategy S1) is applied on the fly
with :func:`build_rain_transform`. Labels are copied unchanged. SHA-256 hashes of every generated
image are written to ``outputs/tables/rain_hashes.csv`` for reproducibility.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import zlib
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import pandas as pd
from tqdm import tqdm

from src.data.split import write_list
from src.utils.common import (
    base_parser,
    imread,
    imwrite,
    load_config,
    project_path,
    setup_logging,
)

log = logging.getLogger("synth_rain")

LEVEL_TO_SET: dict[str, str] = {"drizzle": "B1", "heavy": "B2", "torrential": "B3"}


def build_rain_transform(level_cfg: dict[str, Any], p: float = 1.0) -> A.Compose:
    """Build the albumentations pipeline for one rain level.

    Args:
        level_cfg: One entry of ``rain.yaml: levels``.
        p: Probability of applying the whole pipeline.

    Returns:
        Composed transform (rain streaks -> darkening -> blur).
    """
    rain = dict(level_cfg["rain"])
    rain["slant_range"] = tuple(rain["slant_range"])
    rain["drop_color"] = tuple(rain["drop_color"])
    bc = level_cfg["brightness_contrast"]
    blur = level_cfg["blur"]
    if blur["type"] == "gaussian":
        blur_t = A.GaussianBlur(
            blur_limit=tuple(blur["blur_limit"]), sigma_limit=tuple(blur["sigma_limit"]), p=1.0
        )
    else:
        blur_t = A.MotionBlur(blur_limit=tuple(blur["blur_limit"]), p=1.0)
    return A.Compose(
        [
            A.RandomRain(**rain, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=tuple(bc["brightness_limit"]),
                contrast_limit=tuple(bc["contrast_limit"]),
                p=1.0,
            ),
            blur_t,
        ],
        p=p,
    )


def image_seed(base: int, level_idx: int, stem: str) -> int:
    """Derive a per-image seed that does not depend on processing order.

    Args:
        base: Global seed.
        level_idx: Index of the rain level.
        stem: Image file stem.

    Returns:
        Seed in ``[0, 2**31)``.
    """
    return (base * 1_000_003 + level_idx * 7919 + zlib.crc32(stem.encode())) % (2**31)


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File.

    Returns:
        Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_list(list_path: Path) -> list[Path]:
    """Resolve an Ultralytics ``./``-relative image list.

    Args:
        list_path: ``.txt`` list.

    Returns:
        Absolute image paths.
    """
    lines = list_path.read_text(encoding="utf-8").splitlines()
    return [
        (list_path.parent / ln[2:]).resolve() if ln.startswith("./") else Path(ln)
        for ln in lines
        if ln.strip()
    ]


def count_missing(ref: pd.DataFrame, out_root: Path) -> int:
    """Count images listed in ``rain_hashes.csv`` whose image or label file is missing on disk.

    ``rain_hashes.csv`` is committed but the generated images are not, so on a fresh clone the
    table exists while ``data/processed/rdd_rain/`` is empty.

    Args:
        ref: Contents of the committed ``rain_hashes.csv``.
        out_root: Root of the generated rain folders (``data/processed/rdd_rain``).

    Returns:
        Number of listed images that are missing.
    """
    return sum(
        not (out_root / level / "images" / file).exists()
        or not (out_root / level / "labels" / f"{Path(file).stem}.txt").exists()
        for level, file in zip(ref["level"], ref["file"], strict=True)
    )


def main() -> None:
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    args = parser.parse_args()
    setup_logging("synth_rain")
    cfg = load_config("data")
    rain_cfg = load_config("rain")
    splits = project_path(cfg["paths"]["splits"])
    out_root = project_path(cfg["paths"]["rain"])
    tables = project_path(cfg["paths"]["tables"])
    log.info("LIMITATION: %s", rain_cfg["limitations"])

    images = read_list(splits / "rdd_test.txt")
    if args.quick:  # pipeline check only: separate folder, no lists or hashes
        images = images[: max(1, len(images) // 20)]
        out_root = out_root.with_name(out_root.name + "_quick")
    hash_csv = tables / "rain_hashes.csv"
    ref = None
    if hash_csv.exists() and not args.quick:
        ref = pd.read_csv(hash_csv)
        missing = count_missing(ref, out_root)
        if not missing and not args.force:
            log.warning(
                "%s exists: B1-B3 were already generated. Use --force to rebuild.", hash_csv
            )
            return
        if missing:
            log.info(
                "%s exists but %d/%d listed images are missing under %s (fresh clone?): "
                "regenerating B1-B3",
                hash_csv,
                missing,
                len(ref),
                out_root,
            )
    log.info("Generating %d levels x %d RDD test images", len(rain_cfg["levels"]), len(images))
    if args.dry_run:
        return

    rows = []
    for li, (level, lcfg) in enumerate(rain_cfg["levels"].items()):
        t = build_rain_transform(lcfg)
        img_dir, lbl_dir = out_root / level / "images", out_root / level / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        outs = []
        for src in tqdm(images, desc=level):
            t.set_random_seed(image_seed(rain_cfg["seed"], li, src.stem))
            rgb = cv2.cvtColor(imread(src), cv2.COLOR_BGR2RGB)
            res = cv2.cvtColor(t(image=rgb)["image"], cv2.COLOR_RGB2BGR)
            dst = img_dir / src.name
            imwrite(dst, res, [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy2(src.parent.parent / "labels" / f"{src.stem}.txt", lbl_dir)
            outs.append(dst)
            rows.append(
                {
                    "level": level,
                    "test_set": LEVEL_TO_SET[level],
                    "file": src.name,
                    "sha256": sha256(dst),
                }
            )
        if not args.quick:
            write_list(splits / f"{LEVEL_TO_SET[level]}.txt", outs, splits)
    if args.quick:
        log.info("--quick: hashes not recorded")
        return
    new = pd.DataFrame(rows)
    if ref is not None:  # reproducibility check against the previously recorded hashes
        merged = new.merge(ref, on=["level", "file"], how="outer", suffixes=("", "_ref"))
        n_diff = int((merged["sha256"] != merged["sha256_ref"]).sum())
        if n_diff:
            log.warning(
                "%d/%d images differ from the recorded hashes (different OpenCV/albumentations "
                "version or JPEG encoder?). Inspect with `git diff %s` before committing it.",
                n_diff,
                len(merged),
                hash_csv,
            )
        else:
            log.info("All %d image hashes match the recorded %s", len(new), hash_csv)
    tables.mkdir(parents=True, exist_ok=True)
    new.to_csv(hash_csv, index=False)
    log.info("Wrote %d images and %s", len(rows), hash_csv)


if __name__ == "__main__":
    main()
