"""Create leakage-safe splits and Ultralytics dataset YAMLs.

RDD: per country, consecutive frames (by file number) are grouped into blocks of ``block_size``
and whole blocks go to train/val/test (70/15/15), so near-identical neighbouring frames never
straddle splits. Negatives are then subsampled per split to at most ``neg_ratio``.

MWPD: normal -> test set C, rainy -> test set D (``D_full``). For strategy S3, rainy images are
also split 50/50 into ``mwpd_rainy_train`` / ``D_half`` by near-duplicate groups (dHash).

Split lists are written to ``data/processed/splits/*.txt`` as ``./``-relative image paths.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.common import (
    base_parser,
    imread,
    load_config,
    project_path,
    seed_everything,
    setup_logging,
)

log = logging.getLogger("split")

TEST_SET_LISTS: dict[str, str] = {
    "A": "rdd_test.txt",
    "B1": "B1.txt",
    "B2": "B2.txt",
    "B3": "B3.txt",
    "C": "C.txt",
    "D": "D_full.txt",
    "D_half": "D_half.txt",
}


def block_split(index: pd.DataFrame, ratios: dict[str, float], block: int, seed: int) -> pd.Series:
    """Assign each RDD image to a split by shuffling consecutive-frame blocks per country.

    Blocks are assigned greedily so each split's image count tracks its target ratio.

    Args:
        index: RDD index with ``country`` and ``seq`` columns.
        ratios: Split name -> fraction (sums to 1).
        block: Number of consecutive frames per block.
        seed: Random seed.

    Returns:
        Split name per row (aligned with ``index``).
    """
    rng = np.random.default_rng(seed)
    out = pd.Series(index=index.index, dtype=object)
    names = list(ratios)
    for _, df in index.groupby("country", sort=True):
        order = df.sort_values("seq")
        blocks = (order["seq"] // block).to_numpy()  # by file number, gaps included
        ids = rng.permutation(np.unique(blocks))
        counts = dict.fromkeys(names, 0)
        total = len(order)
        for b in ids:
            members = order.index[blocks == b]
            # pick the split furthest below its target share
            name = min(names, key=lambda n: counts[n] / total - ratios[n])
            out.loc[members] = name
            counts[name] += len(members)
    return out


def subsample_negatives(df: pd.DataFrame, neg_ratio: float, seed: int) -> pd.DataFrame:
    """Keep all positives and at most ``neg_ratio`` negatives (per country) in one split.

    Args:
        df: Rows of one split.
        neg_ratio: Maximum fraction of negatives in the result.
        seed: Random seed.

    Returns:
        Filtered rows.
    """
    keep = []
    for _, g in df.groupby("country", sort=True):
        pos, neg = g[~g["negative"]], g[g["negative"]]
        n_neg = min(len(neg), int(len(pos) * neg_ratio / (1 - neg_ratio)))
        keep += [pos, neg.sample(n=n_neg, random_state=seed)]
    return pd.concat(keep).sort_index()


def dhash(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Compute 64-bit difference hashes of an image and its horizontal flip.

    Args:
        path: Image file.

    Returns:
        ``(hash, flipped_hash)`` as boolean arrays.
    """
    import cv2

    g = imread(path, 0)
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    sf = s[:, ::-1]
    return (s[:, 1:] > s[:, :-1]).ravel(), (sf[:, 1:] > sf[:, :-1]).ravel()


def near_dup_groups(paths: list[Path], thr: int) -> np.ndarray:
    """Group images whose dHash (or flipped dHash) Hamming distance is <= ``thr``.

    Args:
        paths: Image files.
        thr: Hamming-distance threshold.

    Returns:
        Group id per image.
    """
    hs = [dhash(p) for p in paths]
    h = np.array([a for a, _ in hs])
    hf = np.array([b for _, b in hs])
    dist = np.minimum((h[:, None] != h[None]).sum(-1), (h[:, None] != hf[None]).sum(-1))
    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(dist <= thr, k=1)), strict=True):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(paths))])


def write_list(path: Path, images: list[Path], splits_dir: Path) -> None:
    """Write an Ultralytics image list with ``./``-relative paths (relative to the list's folder).

    Args:
        path: Output ``.txt``.
        images: Image files.
        splits_dir: Directory containing the list.
    """
    rel = [Path(p).resolve().relative_to(splits_dir.parent.resolve()).as_posix() for p in images]
    path.write_text("".join(f"./../{r}\n" for r in rel), encoding="utf-8")


def write_yaml(
    path: Path, root: Path, mapping: dict[str, str | list[str]], names: list[str]
) -> None:
    """Write an Ultralytics dataset YAML whose lists live in ``<root>/splits/``.

    ``path`` is absolute because Ultralytics resolves a relative ``path`` against its global
    datasets directory, not the YAML's folder. Re-run ``split.py`` after moving the project.

    Args:
        path: Output YAML.
        root: ``data/processed`` directory.
        mapping: Ultralytics keys (``train``/``val``/``test``) -> list file name(s).
        names: Class names.
    """

    def rel(v: str | list[str]) -> str | list[str]:
        """Prefix list file name(s) with ``splits/``."""
        return [f"splits/{x}" for x in v] if isinstance(v, list) else f"splits/{v}"

    data: dict = {"path": root.resolve().as_posix(), **{k: rel(v) for k, v in mapping.items()}}
    data["names"] = dict(enumerate(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Auto-generated by src/data/split.py\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    args = parser.parse_args()
    setup_logging("split")
    cfg = load_config("data")
    seed = cfg["seed"]
    seed_everything(seed)
    rdd_cfg, mwpd_cfg = cfg["rdd"], cfg["mwpd"]
    proc = project_path(cfg["paths"]["processed"])
    splits_dir = project_path(cfg["paths"]["splits"])
    rdd_dir = project_path(cfg["paths"]["processed_rdd"])
    mwpd_dir = project_path(cfg["paths"]["processed_mwpd"])
    names = cfg["classes"]["names"]

    # ---------------- RDD ----------------
    index = pd.read_csv(rdd_dir / "index.csv")
    index["block"] = index["seq"] // rdd_cfg["block_size"]
    index["split"] = block_split(index, rdd_cfg["split_ratios"], rdd_cfg["block_size"], seed)
    blocks_per_split = index.groupby(["country", "block"])["split"].nunique()
    assert (blocks_per_split == 1).all(), "a frame block was assigned to more than one split"
    selected = pd.concat(
        subsample_negatives(index[index["split"] == s], rdd_cfg["neg_ratio"], seed)
        for s in rdd_cfg["split_ratios"]
    )
    table = selected.pivot_table(
        index="country", columns="split", values="stem", aggfunc="count", fill_value=0
    )[list(rdd_cfg["split_ratios"])]
    share = (table.div(table.sum(axis=1), axis=0) * 100).round(1)
    log.info("RDD images per country x split:\n%s", table.to_string())
    log.info("RDD split share per country (%%):\n%s", share.to_string())
    log.info(
        "RDD negatives per split: %s",
        selected.groupby("split")["negative"].mean().round(3).to_dict(),
    )

    # ---------------- MWPD ----------------
    meta = pd.read_csv(mwpd_dir / "meta.csv")
    img_dir = mwpd_dir / "images"
    normal = meta[meta["weather"] == "normal"]
    rainy = meta[meta["weather"] == "rainy"].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    groups = near_dup_groups([img_dir / f"{s}.jpg" for s in rainy["stem"]], thr=10)
    uniq = rng.permutation(np.unique(groups))
    target, train_groups, n = mwpd_cfg["rainy_train_fraction"] * len(rainy), set(), 0
    for g in uniq:
        if n >= target:
            break
        train_groups.add(g)
        n += int((groups == g).sum())
    rainy["s3"] = np.where(np.isin(groups, list(train_groups)), "rainy_train", "D_half")
    log.info(
        "MWPD: C (normal) %d, D_full (rainy) %d, S3 rainy_train %d / D_half %d",
        len(normal),
        len(rainy),
        (rainy["s3"] == "rainy_train").sum(),
        (rainy["s3"] == "D_half").sum(),
    )

    # ---------------- leakage checks ----------------
    sets = {
        "rdd_train": set(selected.loc[selected["split"] == "train", "stem"]),
        "rdd_val": set(selected.loc[selected["split"] == "val", "stem"]),
        "rdd_test": set(selected.loc[selected["split"] == "test", "stem"]),
        "C": set(normal["stem"]),
        "D_full": set(rainy["stem"]),
        "mwpd_rainy_train": set(rainy.loc[rainy["s3"] == "rainy_train", "stem"]),
        "D_half": set(rainy.loc[rainy["s3"] == "D_half", "stem"]),
    }
    disjoint = [
        ("rdd_train", "rdd_val"),
        ("rdd_train", "rdd_test"),
        ("rdd_val", "rdd_test"),
        ("C", "D_full"),
        ("mwpd_rainy_train", "D_half"),
        ("mwpd_rainy_train", "C"),
    ]
    for a, b in disjoint:
        assert not sets[a] & sets[b], f"split overlap between {a} and {b}"
    log.info("Leakage check passed: %d disjoint split pairs", len(disjoint))

    if args.dry_run:
        return
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name in ("rdd_train", "rdd_val", "rdd_test"):
        write_list(
            splits_dir / f"{name}.txt",
            [rdd_dir / "images" / f"{s}.jpg" for s in sorted(sets[name])],
            splits_dir,
        )
    for name in ("C", "D_full", "mwpd_rainy_train", "D_half"):
        write_list(
            splits_dir / f"{name}.txt",
            [img_dir / f"{s}.jpg" for s in sorted(sets[name])],
            splits_dir,
        )
    assign = pd.concat(
        [
            selected[["stem", "country", "split", "negative"]].assign(dataset="rdd"),
            normal[["stem"]].assign(dataset="mwpd", split="C"),
            rainy[["stem", "s3"]]
            .rename(columns={"s3": "split_s3"})
            .assign(dataset="mwpd", split="D_full"),
        ]
    )
    assign.to_csv(splits_dir / "assignments.csv", index=False)

    rdd_lists = {"train": "rdd_train.txt", "val": "rdd_val.txt", "test": "rdd_test.txt"}
    write_yaml(proc / "rdd.yaml", proc, rdd_lists, names)
    write_yaml(
        proc / "mwpd.yaml",
        proc,
        {"train": "mwpd_rainy_train.txt", "val": "C.txt", "test": "D_full.txt"},
        names,
    )
    # S3 trains on RDD train + MWPD rainy_train (Ultralytics accepts a list of list files).
    write_yaml(
        proc / "rdd_s3.yaml",
        proc,
        {**rdd_lists, "train": ["rdd_train.txt", "mwpd_rainy_train.txt"]},
        names,
    )
    for ts, lst in TEST_SET_LISTS.items():
        write_yaml(proc / "testsets" / f"{ts}.yaml", proc, {"train": lst, "val": lst}, names)
    log.info("Wrote split lists to %s and dataset YAMLs to %s", splits_dir, proc)


if __name__ == "__main__":
    main()
