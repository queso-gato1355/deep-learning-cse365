"""Shared helpers: project paths, config loading, logging and seeding."""

from __future__ import annotations

import argparse
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config from ``configs/``.

    Args:
        name: File stem (e.g. ``"data"``) or file name (``"data.yaml"``).

    Returns:
        Parsed config dictionary.
    """
    path = CONFIG_DIR / (name if name.endswith(".yaml") else f"{name}.yaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(rel: str | Path) -> Path:
    """Resolve a project-relative path from a config value.

    Args:
        rel: Path relative to the project root.

    Returns:
        Absolute path.
    """
    return ROOT / rel


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging to console and ``outputs/logs/{name}_{timestamp}.log``.

    Args:
        name: Log file prefix, usually the script name.
        level: Logging level.

    Returns:
        Logger named ``name``.
    """
    log_dir = project_path(load_config("data")["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"{name}_{stamp}.log", encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger(name)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def base_parser(description: str) -> argparse.ArgumentParser:
    """Create an argument parser with the common ``--quick``/``--dry-run``/``--force`` flags.

    Args:
        description: Script description.

    Returns:
        Parser with shared flags.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true", help="Use 5%% of data / 3 epochs.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; write nothing.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser


def imread(path: Path, flags: int = 1) -> np.ndarray:
    """Read an image; unlike ``cv2.imread`` this works with non-ASCII Windows paths.

    Args:
        path: Image file.
        flags: ``cv2.IMREAD_*`` flag (1 = color BGR, 0 = grayscale).

    Returns:
        Decoded image array.

    Raises:
        OSError: If the file cannot be decoded.
    """
    import cv2

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if img is None:
        raise OSError(f"Cannot read image {path}")
    return img


def imwrite(path: Path, img: np.ndarray, params: list[int] | None = None) -> None:
    """Write an image; unlike ``cv2.imwrite`` this works with non-ASCII Windows paths.

    Args:
        path: Destination file; the extension selects the codec.
        img: Image array.
        params: Optional ``cv2.IMWRITE_*`` parameters.
    """
    import cv2

    ok, buf = cv2.imencode(path.suffix, img, params or [])
    if not ok:
        raise OSError(f"Cannot encode image {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(path)
