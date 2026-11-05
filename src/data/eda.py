"""Exploratory data analysis: figures to ``outputs/figures`` and tables to ``outputs/tables``.

All figures use English labels, the seaborn ``colorblind`` palette, and are saved as 300-dpi PNG
and PDF.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data.synth_rain import read_list  # noqa: E402
from src.utils.common import (  # noqa: E402
    base_parser,
    imread,
    load_config,
    project_path,
    setup_logging,
)

log = logging.getLogger("eda")

IMGSZ = 640
SPLIT_ORDER = ["train", "val", "test"]
COCO_SMALL, COCO_MEDIUM = 32**2, 96**2


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Save a figure as PNG (300 dpi) and PDF, then close it.

    Args:
        fig: Figure.
        out_dir: Output directory.
        name: File stem.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s.{png,pdf}", name)


def load_boxes(images: list[Path], sizes: dict[str, tuple[int, int]] | None = None) -> pd.DataFrame:
    """Load YOLO boxes for a list of images.

    Args:
        images: Image paths (labels are found by replacing ``images`` with ``labels``).
        sizes: Optional ``stem -> (width, height)``; defaults to 640x640.

    Returns:
        One row per box with ``stem, w, h, area_ratio, area_640`` (area in px after letterboxing
        the image to 640 on its long side). Images without boxes get a row with NaN box fields.
    """
    rows = []
    for img in images:
        lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
        W, H = (sizes or {}).get(img.stem, (IMGSZ, IMGSZ))  # noqa: N806
        scale = IMGSZ / max(W, H)
        lines = [r.split() for r in lbl.read_text(encoding="utf-8").splitlines() if r.strip()]
        if not lines:
            rows.append({"stem": img.stem, "w": np.nan, "h": np.nan})
        for p in lines:
            bw, bh = float(p[3]), float(p[4])
            rows.append(
                {"stem": img.stem, "w": bw, "h": bh, "area_640": bw * W * scale * bh * H * scale}
            )
    df = pd.DataFrame(rows)
    df["area_ratio"] = df["w"] * df["h"]
    return df


def draw_boxes(img: np.ndarray, label: Path, color: tuple[int, int, int]) -> np.ndarray:
    """Draw YOLO boxes on a BGR image.

    Args:
        img: BGR image.
        label: YOLO label file.
        color: BGR color.

    Returns:
        Annotated copy.
    """
    out = img.copy()
    h, w = out.shape[:2]
    t = max(2, int(round(max(h, w) / 300)))
    for r in label.read_text(encoding="utf-8").splitlines():
        if not r.strip():
            continue
        _, cx, cy, bw, bh = map(float, r.split())
        p0 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
        p1 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
        cv2.rectangle(out, p0, p1, color, t)
    return out


def is_augmented(path: Path) -> bool:
    """Heuristically flag Roboflow-augmented copies (rotation leaves black corners).

    Used only to choose representative examples for figures.

    Args:
        path: Image file.

    Returns:
        True if at least two corners are (near-)black.
    """
    g = imread(path, 0)
    corners = [g[:10, :10], g[:10, -10:], g[-10:, :10], g[-10:, -10:]]
    return sum(float(c.mean()) < 8 for c in corners) >= 2


def square(img: np.ndarray, size: int = 480) -> np.ndarray:
    """Center-crop to a square and resize, for tidy grids.

    Args:
        img: Image array.
        size: Output side.

    Returns:
        Square image.
    """
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y0 : y0 + s, x0 : x0 + s], (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:  # noqa: PLR0915
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    args = parser.parse_args()
    setup_logging("eda")
    cfg = load_config("data")
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    sns.set_theme(style="whitegrid", palette="colorblind", context="paper")
    pal = sns.color_palette("colorblind")
    figs, tables = project_path(cfg["paths"]["figures"]), project_path(cfg["paths"]["tables"])
    splits = project_path(cfg["paths"]["splits"])
    rdd_dir = project_path(cfg["paths"]["processed_rdd"])

    index = pd.read_csv(rdd_dir / "index.csv").set_index("stem")
    sizes = {s: (int(r.width), int(r.height)) for s, r in index.iterrows()}
    lists = {
        n: read_list(splits / f"{n}.txt")
        for n in [
            "rdd_train",
            "rdd_val",
            "rdd_test",
            "C",
            "D_full",
            "mwpd_rainy_train",
            "D_half",
            "B1",
            "B2",
            "B3",
        ]
    }
    boxes = {
        n: load_boxes(v, sizes if n.startswith(("rdd", "B")) else None) for n, v in lists.items()
    }
    if args.dry_run:
        log.info("Loaded %s", {k: len(v) for k, v in lists.items()})
        return

    # ---- fig_rdd_country_counts ----
    rows = []
    for split in SPLIT_ORDER:
        b = boxes[f"rdd_{split}"]
        per_img = b.groupby("stem")["w"].count()
        for stem, n in per_img.items():
            rows.append({"split": split, "country": index.loc[stem, "country"], "boxes": n})
    cc = pd.DataFrame(rows)
    countries = cfg["rdd"]["countries"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, (col, title) in zip(
        axes, [("count", "Images"), ("boxes", "D40 (pothole) boxes")], strict=True
    ):
        agg = cc.groupby(["country", "split"])["boxes"].agg(["count", "sum"])
        agg = agg["count" if col == "count" else "sum"].unstack()[SPLIT_ORDER].reindex(countries)
        agg.plot(kind="bar", stacked=True, ax=ax, color=pal[:3], width=0.75)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.set_xticklabels([c.replace("_", " ") for c in countries], rotation=30, ha="right")
        ax.legend(title="Split")
    fig.suptitle("RDD2022 subset used (China_Drone excluded)")
    save(fig, figs, "fig_rdd_country_counts")

    # ---- fig_boxes_per_image ----
    groups = {
        "RDD2022 (all splits)": pd.concat([boxes[f"rdd_{s}"] for s in SPLIT_ORDER]),
        "MWPD normal (C)": boxes["C"],
        "MWPD rainy (D)": boxes["D_full"],
    }
    rows = []
    for name, b in groups.items():
        counts = b.groupby("stem")["w"].count().clip(upper=10)
        frac = counts.value_counts(normalize=True).sort_index()
        rows += [{"dataset": name, "n": int(k), "fraction": v} for k, v in frac.items()]
    bpi = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    sns.barplot(data=bpi, x="n", y="fraction", hue="dataset", ax=ax)
    ax.set_xticks(range(11))
    ax.set_xticklabels([str(i) for i in range(10)] + ["10+"])
    ax.set_xlabel("Potholes per image")
    ax.set_ylabel("Fraction of images")
    ax.legend(title="")
    save(fig, figs, "fig_boxes_per_image")

    # ---- fig_box_size ----
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for (name, b), c in zip(groups.items(), pal, strict=False):
        a = b["area_640"].dropna()
        sns.kdeplot(
            np.log10(a[a > 0]),
            ax=ax,
            label=f"{name} (n={len(a)})",
            color=c,
            fill=True,
            alpha=0.2,
            common_norm=False,
        )
    for v, lab in [(COCO_SMALL, "small | medium\n(32²)"), (COCO_MEDIUM, "medium | large\n(96²)")]:
        ax.axvline(np.log10(v), color="k", ls="--", lw=1)
        ax.text(
            np.log10(v) + 0.05,
            0.02,
            lab,
            ha="left",
            va="bottom",
            fontsize=7,
            transform=ax.get_xaxis_transform(),
            bbox={"fc": "white", "ec": "none", "alpha": 0.8},
        )
    ax.set_xlabel("Box area at 640-px letterbox (log$_{10}$ px²)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, figs, "fig_box_size")

    # ---- fig_samples_grid ----
    def pick(paths: list[Path], k: int) -> list[Path]:
        pos = [
            p
            for p in paths
            if (p.parent.parent / "labels" / f"{p.stem}.txt").stat().st_size and not is_augmented(p)
        ]
        return [pos[i] for i in rng.choice(len(pos), size=k, replace=False)]

    rows_def = [
        ("RDD2022", pick(lists["rdd_test"], 4)),
        ("MWPD normal", pick(lists["C"], 4)),
        ("MWPD rainy", pick(lists["D_full"], 4)),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.8))
    for r, (name, paths) in enumerate(rows_def):
        for c, p in enumerate(paths):
            img = draw_boxes(imread(p), p.parent.parent / "labels" / f"{p.stem}.txt", (0, 255, 0))
            axes[r, c].imshow(cv2.cvtColor(square(img), cv2.COLOR_BGR2RGB))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=11)
    fig.tight_layout()
    save(fig, figs, "fig_samples_grid")

    # ---- fig_rain_levels ----
    src = pick(lists["rdd_test"], 1)[0]
    rain_root = project_path(cfg["paths"]["rain"])
    panels = (
        [("Clear (A)", src)]
        + [
            (f"{lvl.capitalize()} ({ts})", rain_root / lvl / "images" / src.name)
            for lvl, ts in [("drizzle", "B1"), ("heavy", "B2"), ("torrential", "B3")]
        ]
        + [("Real rain, MWPD (D)", pick(lists["D_full"], 1)[0])]
    )
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
    for ax, (title, p) in zip(axes, panels, strict=True):
        ax.imshow(cv2.cvtColor(square(imread(p)), cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    save(fig, figs, "fig_rain_levels")

    # ---- tab_dataset_summary ----
    def summarize(name: str, split: str, b: pd.DataFrame, paths: list[Path], rdd: bool) -> dict:
        res = [sizes[p.stem] for p in paths] if rdd else [(640, 640)] * len(paths)
        return {
            "Dataset": name,
            "Split": split,
            "Images": len(paths),
            "Boxes": int(b["w"].notna().sum()),
            "Neg. images": int(b.groupby("stem")["w"].count().eq(0).sum()),
            "Mean box area (%)": round(100 * b["area_ratio"].mean(), 2),
            "Resolution": f"{min(w for w, _ in res)}x{min(h for _, h in res)}"
            f" – {max(w for w, _ in res)}x{max(h for _, h in res)}",
        }

    summ = pd.DataFrame(
        [
            summarize("RDD2022", "train", boxes["rdd_train"], lists["rdd_train"], True),
            summarize("RDD2022", "val", boxes["rdd_val"], lists["rdd_val"], True),
            summarize("RDD2022", "test (A)", boxes["rdd_test"], lists["rdd_test"], True),
            summarize("MWPD", "normal (C)", boxes["C"], lists["C"], False),
            summarize("MWPD", "rainy (D_full)", boxes["D_full"], lists["D_full"], False),
            summarize(
                "MWPD",
                "rainy_train (S3)",
                boxes["mwpd_rainy_train"],
                lists["mwpd_rainy_train"],
                False,
            ),
            summarize("MWPD", "rainy (D_half, S3)", boxes["D_half"], lists["D_half"], False),
        ]
    )
    write_table(
        summ, tables, "tab_dataset_summary", "Dataset and split statistics (pothole class only)."
    )

    # ---- tab_test_sets ----
    ts = pd.DataFrame(
        [
            {
                "ID": "A",
                "Source": "RDD2022 own test split",
                "Condition": "clear",
                "Images": len(lists["rdd_test"]),
                "Purpose": "baseline",
            },
            *[
                {
                    "ID": f"B{i}",
                    "Source": "A + synthetic rain",
                    "Condition": lvl,
                    "Images": len(lists[f"B{i}"]),
                    "Purpose": "pure rain effect (A→B)",
                }
                for i, lvl in enumerate(["drizzle", "heavy", "torrential"], start=1)
            ],
            {
                "ID": "C",
                "Source": "MWPD",
                "Condition": "normal weather",
                "Images": len(lists["C"]),
                "Purpose": "domain gap (A→C)",
            },
            {
                "ID": "D",
                "Source": "MWPD",
                "Condition": "real rain",
                "Images": len(lists["D_full"]),
                "Purpose": "real rain effect (C→D)",
            },
        ]
    )
    write_table(ts, tables, "tab_test_sets", "Fixed test sets used in all experiments.")

    log.info("Dataset summary:\n%s", summ.to_string(index=False))
    log.info("Test sets:\n%s", ts.to_string(index=False))

    # ---- LaTeX macros so slides/report never hand-copy numbers ----
    pos = index[~index["negative"]]
    top2 = pos["country"].value_counts().head(2)
    small = {n: float((groups[n]["area_640"].dropna() < COCO_SMALL).mean()) for n in groups}
    macros = {
        "nRddAnnotated": len(index),
        "nRddPositive": len(pos),
        "pctRddPositive": round(100 * len(pos) / len(index)),
        "nRddBoxes": int(pos["n_boxes"].sum()),
        "pctRddTopTwo": round(100 * top2.sum() / len(pos)),
        "nRddTrain": len(lists["rdd_train"]),
        "nRddVal": len(lists["rdd_val"]),
        "nA": len(lists["rdd_test"]),
        "nB": len(lists["B1"]),
        "nC": len(lists["C"]),
        "nD": len(lists["D_full"]),
        "nMwpd": len(lists["C"]) + len(lists["D_full"]),
        "nMwpdRainyTrain": len(lists["mwpd_rainy_train"]),
        "nDHalf": len(lists["D_half"]),
        "nBoxA": int(boxes["rdd_test"]["w"].notna().sum()),
        "nBoxC": int(boxes["C"]["w"].notna().sum()),
        "nBoxD": int(boxes["D_full"]["w"].notna().sum()),
        "pctSmallRdd": round(100 * small["RDD2022 (all splits)"]),
        "pctSmallMwpd": round(
            100
            * (
                small["MWPD normal (C)"] * len(lists["C"])
                + small["MWPD rainy (D)"] * len(lists["D_full"])
            )
            / (len(lists["C"]) + len(lists["D_full"]))
        ),
        "blockSize": cfg["rdd"]["block_size"],
        "negRatioPct": round(100 * cfg["rdd"]["neg_ratio"]),
    }
    write_macros(macros, tables / "data_stats.tex")


def write_macros(values: dict[str, int | float], path: Path) -> None:
    r"""Write ``\newcommand`` macros (thousands separated with ``{,}``) for LaTeX documents.

    Args:
        values: Macro name (letters only) -> number.
        path: Output ``.tex`` file.
    """
    lines = ["% Auto-generated by src/data/eda.py -- do not edit by hand."]
    for k, v in values.items():
        txt = f"{v:,}".replace(",", "{,}") if isinstance(v, int) else f"{v:g}"
        lines.append(f"\\newcommand{{\\{k}}}{{{txt}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved %s (%d macros)", path.name, len(values))


def write_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str) -> None:
    """Write a table as CSV and a booktabs LaTeX fragment.

    Args:
        df: Table.
        out_dir: Output directory.
        name: File stem (also used as LaTeX label ``tab:<name>``).
        caption: LaTeX caption.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    tex = df.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=f"tab:{name}",
        position="t",
        column_format="l" * len(df.columns),
    )
    tex = tex.replace("–", "--").replace("→", r"$\to$")
    (out_dir / f"{name}.tex").write_text(tex, encoding="utf-8")
    log.info("Saved %s.{csv,tex}", name)


if __name__ == "__main__":
    main()
