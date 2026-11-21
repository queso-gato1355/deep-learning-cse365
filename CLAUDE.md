# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Course project (CSE36501 Deep Learning, UNIST, Fall 2026): lightweight pothole detection for
vehicle-based road monitoring (refocused in commit 662c326; it was originally "rain-robust pothole detection").
- RQ1: accuracy–efficiency trade-off: how small/fast can the detector be while staying useful?
- RQ2: which efficiency lever costs the least accuracy: smaller model, lower input resolution, lightweight module, reduced precision?
- RQ3: does an efficient model stay reliable across domains (other countries/cameras) and in rain?

Every model is reported as accuracy (mAP50, mAP50-95, recall, small-object AP) vs. cost (params,
GFLOPs, FPS on GPU and CPU) on test sets A, B1–B3, C, D.

Phase 1 (data pipeline + EDA) and Phase 1.5 (proposal slides) are done. `src/train`, `src/eval`,
`src/models`, `src/report` are empty stubs for the upcoming phases. Note: `configs/experiments.yaml`
still describes the old rain-strategy matrix (S0–S3, E1–E4) and has not been updated for the new RQs.

## Course requirements

The official spec is `2026_DL_Project.pdf`; a Korean summary is in `project_guidelines.md` (untracked). Key points:
- **All deliverables in English** (slides, video scripts, code comments, README, emails). Discussion with the user may be in Korean.
- Grading emphasizes effort, clarity and depth of understanding over raw numbers: always be able to explain *why* something worked or failed.
- Deadlines (KST, no late submissions): Proposal 2026-11-06 (slides PDF + ≤5 min video, covers Problem + Dataset only); **Final 2026-12-13** (slides PDF + ≤15 min detailed video + source code); in-class ≤5 min talk 12-15 / 12-17.
- Final grading (30): Problem & Dataset 8, Model Design & Experiments 10, Results & Discussion 7, Presentation 5. Prioritize experiment design/baseline comparison and result interpretation.
- Required: a baseline compared against improvements, fixed seeds, recorded hyperparameters/environment, explicit splits, results accumulated in tables, and saved qualitative material (success/failure examples, learning curves, error analysis) during experiments. Keep training budgets small enough to run several experiments before 12-13.
- Cite all datasets, code, papers and tutorials on the slides.
- Submission naming: `Proposal_[student IDs].zip`, `Final_[student IDs].zip` (IDs concatenated with `_`).

## Commands

Python 3.11 only (`requires-python = ">=3.11,<3.12"`), venv managed with `uv`. The Makefile picks
`.venv/Scripts/python.exe` on Windows and `.venv/bin/python` elsewhere.

```bash
make env        # uv venv + install requirements.txt (torch cu128 wheels)
make lint       # ruff format src && ruff check src
make download   # fetch MWPD (SHA-256 verified), verify+extract RDD2022 zip (must be downloaded manually first into data/raw/_downloads/rdd2022/)
make data       # full Phase 1 chain: inspect -> convert -> split -> rain -> eda
make tag        # rebuild outputs/tag_weather.html (manual MWPD weather-tag review page)
make proposal   # latexmk slides/proposal.tex and slides/proposal-notes.tex
make kernel     # register .venv as Jupyter kernel "pothole-venv"
make notebook   # regenerate notebooks/01_model_overview.ipynb from build_01_model_overview.py and execute it
make split ARGS='--dry-run'   # ARGS is forwarded to every Phase-1 script
```

Each pipeline step is a module run as `python -m src.data.<step>`, so run single steps the same way
from the repo root. There is no test suite.

## Architecture and conventions

- **Config-driven.** All paths, seeds, split ratios, rain levels and the experiment matrix live in
  `configs/*.yaml`, loaded via `src.utils.common.load_config("data")`. Paths in configs are
  project-root-relative; resolve them with `project_path()`. Don't hardcode paths.
  - `data.yaml`: paths, dataset sources/checksums, RDD countries (China_Drone excluded), `source_class: D40`, split params, seed 42.
  - `rain.yaml`: albumentations params for drizzle/heavy/torrential (= test sets B1/B2/B3).
  - `train.yaml`: shared Ultralytics hyperparameters, `p_rain` for S1, `quick` mode settings.
  - `experiments.yaml`: models (yolov8n/s, rtdetr-l, yolov8s_cbam), strategies S0–S3, test sets A/B1/B2/B3/C/D, experiments E1–E4. `run_id = {model}_{strategy}_s{seed}`.
- **Every script** uses `base_parser()` (shared `--quick`, `--dry-run`, `--force` flags),
  `setup_logging(name)` (console + `outputs/logs/{name}_{timestamp}.log`) and `seed_everything()`.
  Scripts should be idempotent: skip existing outputs unless `--force`, write nothing on `--dry-run`.
- **Use `imread`/`imwrite` from `src.utils.common`**, not `cv2.imread/imwrite` directly; they handle
  non-ASCII Windows paths.
- **Single class.** RDD2022 VOC labels are converted to YOLO with only D40 → class 0 (`pothole`).
- **Leakage-safe splits** (`split.py`): RDD frames are grouped into blocks of 50 consecutive frames per
  country before the 70/15/15 split; negatives capped at `neg_ratio` per split. MWPD normal → test set C,
  rainy → D (`D_full`); for S3 rainy is split 50/50 into `mwpd_rainy_train` / `D_half` by dHash
  near-duplicate groups. `split.py` also writes Ultralytics dataset YAMLs (`rdd.yaml`, `mwpd.yaml`,
  `rdd_s3.yaml`, `testsets/*.yaml`) into `data/processed/`.
- **Synthetic rain** (`synth_rain.py`): B1–B3 are generated once from RDD test (A) with fixed seeds and
  hashed into `outputs/tables/rain_hashes.csv`. For S1 training, use `build_rain_transform()` on the fly
  (train split only).
- MWPD has no weather labels; manual normal/rainy + lighting tags are in
  `data/processed/mwpd/weather_tags.csv`.

## What is committed vs. generated

Images and raw/processed data are gitignored (size/licensing). Only the split lists
(`data/processed/splits/*.txt`, `assignments.csv`) and `weather_tags.csv` are committed, so splits are
reproducible across machines; don't regenerate them casually. `outputs/figures`, `outputs/tables`
(incl. `data_stats.tex` LaTeX macros consumed by the slides) and `outputs/env` are committed;
`outputs/logs`, `outputs/runs`, and model weights are not.

## Style

Ruff, line length 100, Google-style docstrings, type annotations required (rules `E,F,W,I,N,UP,B,D,ANN`).
`import albumentations as A` is allowed (N812 ignored).
