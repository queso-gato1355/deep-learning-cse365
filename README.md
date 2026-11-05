# Lightweight Pothole Detection for Vehicle-Based Road Monitoring

CSE36501 Deep Learning, UNIST, Fall 2026.

Potholes are currently found by road-agency inspections or citizen reports, so small potholes
and those on rural roads are reported late or never. Autonomous vehicles could detect potholes
and report them automatically, but the detector must be cheap enough to run alongside the
driving stack. This project asks whether such an efficient detector can be built.

- **RQ1.** How small and fast can a pothole detector be while keeping useful accuracy? (accuracy–efficiency trade-off)
- **RQ2.** Which efficiency lever costs the least accuracy: a smaller model, a lower input resolution, a lightweight module design, or reduced precision?
- **RQ3.** Does an efficient model stay reliable in conditions a car will meet: other countries and cameras, and rain?

Every model is reported as accuracy (mAP50, mAP50-95, recall, small-object AP) versus cost
(params, GFLOPs, FPS on GPU and CPU). Test sets: A (RDD2022 test, main), B1–B3 (A with
synthetic rain), C (MWPD normal), D (MWPD rainy).

**Status:** Phase 1 (data + EDA) and Phase 1.5 (proposal slides) are complete. Training, evaluation and the report follow.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # .venv/bin/python on Linux/macOS
```

## Data

Raw and processed images are not in this repository.

| Dataset | Source | License |
|---|---|---|
| RDD2022 | [FigShare 21431547](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) (`RDD2022_released_through_CRDDC2022.zip`, md5 `b62bd51d2ffcfaa76c60f234f0cc2bb3`) | CC BY 4.0 |
| MWPD | [Mendeley Data, doi:10.17632/s5hx9n2jc3.2](https://data.mendeley.com/datasets/s5hx9n2jc3/2) | CC BY 4.0 |

1. Download the RDD2022 zip into `data/raw/_downloads/rdd2022/`.
2. `make download` fetches MWPD (SHA-256 verified) and verifies and extracts RDD2022.
3. `make data` runs inspect, convert, split, synthetic rain and EDA.

MWPD has no weather labels. The manual normal/rainy and lighting tags are in
`data/processed/mwpd/weather_tags.csv`; `make tag` rebuilds the review page
`outputs/tag_weather.html`. The split lists under `data/processed/splits/` are committed so splits can be compared across machines.

## Layout

```
configs/   data, rain, train and experiment settings
src/data/  download, inspect, tag_weather, convert_rdd, convert_mwpd, split, synth_rain, eda
outputs/   figures/, tables/ (incl. data_stats.tex macros), env/
slides/    proposal.tex (+ proposal-notes.tex for speaker notes)
```
