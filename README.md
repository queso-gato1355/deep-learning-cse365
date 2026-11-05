# Rain-Robust Pothole Detection

CSE36501 Deep Learning, UNIST, Fall 2026.

- **RQ1.** How much does a pothole detector trained only on clear-weather images degrade under rain?
- **RQ2.** Can synthetic rain overlays serve as a proxy for real rainy conditions?
- **RQ3.** Which strategy improves rain robustness most: (a) synthetic-rain augmentation, (b) architectural change (attention module), or (c) test-time deraining?

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
