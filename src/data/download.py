"""Download raw datasets into ``data/raw/`` (RDD2022 from FigShare, MWPD from Mendeley Data)."""

from __future__ import annotations

import hashlib
import logging
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.utils.common import base_parser, load_config, project_path, setup_logging

log = logging.getLogger("download")


def file_hash(path: Path, algo: str) -> str:
    """Compute a file digest.

    Args:
        path: File to hash.
        algo: ``hashlib`` algorithm name.

    Returns:
        Hex digest.
    """
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected: str | None = None, algo: str = "sha256") -> None:
    """Download ``url`` to ``dest`` (resumable skip if hash already matches).

    Args:
        url: Source URL.
        dest: Destination file.
        expected: Expected hex digest, or ``None`` to skip verification.
        algo: Hash algorithm for ``expected``.

    Raises:
        ValueError: If the downloaded file's digest does not match.
    """
    if dest.exists() and (expected is None or file_hash(dest, algo) == expected):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            shutil.copyfileobj(r.raw, f, length=1 << 20)
    if expected is not None and file_hash(tmp, algo) != expected:
        tmp.unlink()
        raise ValueError(f"Hash mismatch for {dest.name}")
    tmp.replace(dest)


def list_mendeley(api: str, version: int) -> list[dict[str, Any]]:
    """List all files of a Mendeley Data dataset with their folder paths.

    Args:
        api: Dataset public API base URL.
        version: Dataset version.

    Returns:
        File records with an added ``rel_path`` key.
    """
    folders = requests.get(f"{api}/folders/{version}", timeout=60).json()
    by_id = {f["id"]: f for f in folders}

    def folder_path(fid: str) -> Path:
        parts: list[str] = []
        while fid in by_id:
            parts.append(by_id[fid]["name"])
            fid = by_id[fid].get("parent_id", "")
        return Path(*reversed(parts))

    records: list[dict[str, Any]] = []
    for fid in ["root", *by_id]:
        files = requests.get(
            f"{api}/files", params={"folder_id": fid, "version": version}, timeout=60
        ).json()
        for rec in files:
            base = Path() if fid == "root" else folder_path(fid)
            rec["rel_path"] = (base / rec["filename"]).as_posix()
            records.append(rec)
    return records


def download_mwpd(dry_run: bool) -> None:
    """Download every MWPD file into ``data/raw/mwpd/`` and verify SHA-256 hashes.

    Args:
        dry_run: Only list files.
    """
    cfg = load_config("data")
    src = cfg["sources"]["mwpd"]
    out = project_path(cfg["paths"]["raw_mwpd"])
    records = list_mendeley(src["api"], src["version"])
    total = sum(r["size"] for r in records)
    log.info("MWPD: %d files, %.1f MB", len(records), total / 1e6)
    if dry_run:
        return
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [
            pool.submit(
                fetch,
                rec["content_details"]["download_url"],
                out / rec["rel_path"],
                rec["content_details"]["sha256_hash"],
            )
            for rec in records
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="MWPD"):
            fut.result()
    log.info("MWPD downloaded to %s", out)


def extract_rdd(dry_run: bool) -> None:
    """Verify the RDD2022 zip (downloaded with curl) and extract it into ``data/raw/rdd2022/``.

    Everything is extracted unchanged (including excluded countries); filtering happens in
    ``convert_rdd.py`` so the raw copy stays a faithful mirror of the release.

    Args:
        dry_run: Only verify and list.
    """
    cfg = load_config("data")
    src = cfg["sources"]["rdd2022"]
    zpath = project_path(cfg["paths"]["downloads"]) / "rdd2022" / src["filename"]
    out = project_path(cfg["paths"]["raw_rdd"])
    if not zpath.exists():
        log.error("Missing %s. Download it from %s first.", zpath, src["url"])
        return
    digest = file_hash(zpath, "md5")
    if digest != src["md5"]:
        raise ValueError(f"RDD2022 md5 mismatch: {digest} != {src['md5']}")
    log.info("RDD2022 zip md5 verified")
    if dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    extract_nested(zpath, out)
    log.info("RDD2022 extracted to %s", out)


def extract_nested(zpath: Path, out: Path) -> None:
    """Extract a zip, recursively extracting inner zips into sibling folders.

    Args:
        zpath: Zip file.
        out: Destination directory.
    """
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(out)
    for inner in sorted(out.rglob("*.zip")):
        log.info("Extracting inner archive %s", inner.name)
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(inner.parent)
        inner.unlink()


def main() -> None:
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    parser.add_argument("--dataset", choices=["mwpd", "rdd", "all"], default="all")
    args = parser.parse_args()
    setup_logging("download")
    if args.dataset in ("mwpd", "all"):
        download_mwpd(args.dry_run)
    if args.dataset in ("rdd", "all"):
        extract_rdd(args.dry_run)


if __name__ == "__main__":
    main()
