"""Prepare a fresh Google Colab runtime: Drive links, pre-built data, dataset YAML paths.

Colab-only. Mount Drive in the notebook first (``drive.mount`` needs the interactive prompt), then
run from the repo root:
    python -m src.data.colab_setup

Steps (each one is idempotent):
    1. Install packages Colab does not ship (``ultralytics``, ``albumentations``), pinned to
       ``requirements.txt``. Colab's own torch build is kept.
    2. Symlink ``outputs/logs`` and ``outputs/runs`` to ``MyDrive/<drive-dir>/outputs/`` so logs
       and training runs survive the runtime.
    3. Copy ``MyDrive/<drive-dir>/data.zip`` to local disk and extract it into ``data/`` (local
       disk is much faster than Drive for many small files). Skipped once ``data/.unzipped``
       exists, i.e. once per runtime.
    4. Restore the committed split lists (the archive's copies have CRLF line endings).
    5. Point the ``path:`` of every dataset YAML under ``data/processed`` to this machine.
       ``split.py`` writes absolute paths, so YAMLs built on another computer do not work as-is.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from src.utils.common import base_parser, load_config, project_path, setup_logging

log = logging.getLogger("colab_setup")

DRIVE_MYDRIVE = Path("/content/drive/MyDrive")
LOCAL_ZIP = Path("/content/data.zip")
PACKAGES = ("ultralytics", "albumentations")
PATH_LINE = re.compile(r"^path:.*$", re.M)


def sh(cmd: list[str]) -> None:
    """Run a command, raising if it fails.

    Args:
        cmd: Command and arguments.
    """
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pinned(packages: tuple[str, ...]) -> list[str]:
    """Return ``name==version`` pins for ``packages`` from ``requirements.txt``.

    Args:
        packages: Package names to look up.

    Returns:
        Requirement strings, in the order of ``packages``.
    """
    reqs = project_path("requirements.txt").read_text(encoding="utf-8").splitlines()
    pins = {
        r.split("==")[0].strip(): r.strip() for r in reqs if "==" in r and not r.startswith("#")
    }
    return [pins.get(p, p) for p in packages]


def link(target: Path, link_path: Path, dry_run: bool) -> None:
    """Replace ``link_path`` with a symlink to ``target`` (created if missing).

    Args:
        target: Directory on Drive.
        link_path: Path inside the repo.
        dry_run: Only log what would happen.
    """
    log.info("link %s -> %s", link_path, target)
    if dry_run:
        return
    target.mkdir(parents=True, exist_ok=True)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        shutil.rmtree(link_path)
    os.symlink(target, link_path)


def extract(zip_path: Path, data_dir: Path, force: bool, dry_run: bool) -> None:
    """Copy ``zip_path`` to local disk and extract it into ``data_dir``.

    Works whether the archive has a top-level ``data/`` folder or not.

    Args:
        zip_path: ``data.zip`` on Drive.
        data_dir: The repo's ``data/`` directory.
        force: Extract again even if the marker file exists.
        dry_run: Only log what would happen.
    """
    marker = data_dir / ".unzipped"
    if marker.exists() and not force:
        log.info("%s exists: data.zip already extracted in this runtime, skipped.", marker)
        return
    log.info("copy %s (%.2f GB) -> %s", zip_path, zip_path.stat().st_size / 1e9, LOCAL_ZIP)
    if dry_run:
        return
    shutil.copyfile(zip_path, LOCAL_ZIP)
    with zipfile.ZipFile(LOCAL_ZIP) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]
    dest = data_dir.parent if all(n.startswith("data/") for n in names) else data_dir
    dest.mkdir(parents=True, exist_ok=True)
    log.info("extract %d entries into %s", len(names), dest)
    sh(["unzip", "-q", "-o", str(LOCAL_ZIP), "-x", "__MACOSX/*", "-d", str(dest)])
    LOCAL_ZIP.unlink()
    marker.touch()


def fix_yaml_paths(processed: Path, dry_run: bool) -> None:
    """Set ``path:`` in every dataset YAML under ``processed`` to ``processed`` on this machine.

    All YAMLs written by ``split.py``/``road_crop.py`` use ``data/processed`` as their root.

    Args:
        processed: The ``data/processed`` directory.
        dry_run: Only log what would change.
    """
    new_line = f"path: {processed.resolve().as_posix()}"
    for yml in sorted(processed.rglob("*.yaml")):
        text = yml.read_text(encoding="utf-8")
        m = PATH_LINE.search(text)
        if not m or m.group(0).strip() == new_line:
            continue
        log.info("%s: %r -> %r", yml.relative_to(processed), m.group(0).strip(), new_line)
        if not dry_run:
            yml.write_text(PATH_LINE.sub(new_line, text, count=1), encoding="utf-8")


def main() -> None:
    """Run the Colab setup steps."""
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--drive-dir", default="cse365_pothole", help="Folder under MyDrive.")
    parser.add_argument("--zip", default="data.zip", help="Archive name inside --drive-dir.")
    parser.add_argument("--skip-pip", action="store_true", help="Do not install packages.")
    args = parser.parse_args()

    drive_root = DRIVE_MYDRIVE / args.drive_dir
    zip_path = drive_root / args.zip
    if not DRIVE_MYDRIVE.exists():
        sys.exit("Google Drive is not mounted: run drive.mount('/content/drive') in the notebook.")
    if not zip_path.exists():
        sys.exit(f"{zip_path} not found; check --drive-dir / --zip.")

    for name in ("logs", "runs"):
        link(drive_root / "outputs" / name, project_path(f"outputs/{name}"), args.dry_run)
    setup_logging("colab_setup")  # after linking, so the log file lands on Drive

    if not (args.skip_pip or args.dry_run):
        sh([sys.executable, "-m", "pip", "install", "-q", *pinned(PACKAGES)])

    cfg = load_config("data")
    extract(zip_path, project_path("data"), args.force, args.dry_run)
    if not args.dry_run:
        sh(["git", "checkout", "--", cfg["paths"]["splits"]])
    fix_yaml_paths(project_path(cfg["paths"]["processed"]), args.dry_run)

    changed = subprocess.run(
        ["git", "status", "--porcelain", cfg["paths"]["splits"]],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if changed:
        log.warning("Split lists differ from the committed ones:\n%s", changed)
    log.info("Colab setup done.")


if __name__ == "__main__":
    main()
