# Lightweight pothole detection. Run `make help` for targets.
ifeq ($(OS),Windows_NT)
PY ?= deep_learning/Scripts/python.exe
else
PY ?= deep_learning/bin/python
endif
ARGS ?=
# PyTorch wheel index: cu128 (NVIDIA driver >= 570), cu126 (older driver) or cpu
TORCH ?= cu128
EXP ?= E1

.PHONY: help env lint download inspect tag convert split rain eda data crop proposal kernel notebook notebooks train train-list

help:
	@echo "make env       - create deep_learning (uv, Python 3.14, with pip), install torch (TORCH=cu128|cu126|cpu) + requirements"
	@echo "make lint      - ruff format + lint"
	@echo "make download  - fetch MWPD, verify+extract RDD2022 (zip downloaded beforehand)"
	@echo "make data      - Phase 1: inspect, convert, split, synthetic rain, EDA"
	@echo "make crop      - build the road-region (bottom 60%) dataset for E3/E4"
	@echo "make train-list EXP=E1  - show Phase 2 runs and time estimate"
	@echo "make train EXP=E1       - train (finished runs skipped); ARGS='--seeds 42' etc."
	@echo "make proposal  - build slides/proposal.pdf and slides/proposal-notes.pdf (speaker notes)"
	@echo "make kernel    - register deep_learning as Jupyter kernel 'deep_learning'"
	@echo "make notebook  - rebuild and execute notebooks/01_model_overview.ipynb"
	@echo "make notebooks - regenerate notebooks/02_phase2_training.ipynb (saved without outputs)"
	@echo "make tag       - rebuild outputs/tag_weather.html (manual MWPD weather tags)"
	@echo "ARGS='--dry-run' or ARGS='--force' are passed to every Phase-1 script"

env:
	uv venv --python 3.14 --seed --prompt deep_learning deep_learning
	uv pip install --python $(PY) torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/$(TORCH)
	uv pip install --python $(PY) -r requirements.txt

lint:
	$(PY) -m ruff format src
	$(PY) -m ruff check src

download:
	$(PY) -m src.data.download --dataset all $(ARGS)

inspect:
	$(PY) -m src.data.inspect $(ARGS)

tag:
	$(PY) -m src.data.tag_weather $(ARGS)

convert: inspect
	$(PY) -m src.data.convert_rdd $(ARGS)
	$(PY) -m src.data.convert_mwpd $(ARGS)

split: convert
	$(PY) -m src.data.split $(ARGS)

rain: split
	$(PY) -m src.data.synth_rain $(ARGS)

eda: rain
	$(PY) -m src.data.eda $(ARGS)

data: eda

proposal:
	cd slides && latexmk -pdf -interaction=nonstopmode proposal.tex proposal-notes.tex

kernel:
	$(PY) -m ipykernel install --user --name deep_learning --display-name "Python 3.14 (deep_learning)"

notebook:
	$(PY) notebooks/build_01_model_overview.py
	cd notebooks && ../$(PY) -m jupyter nbconvert --to notebook --execute --inplace 01_model_overview.ipynb

crop:
	$(PY) -m src.data.road_crop $(ARGS)

train-list:
	$(PY) -m src.train.train --exp $(EXP) --list $(ARGS)

train:
	$(PY) -m src.train.train --exp $(EXP) --yes $(ARGS)

notebooks:
	$(PY) notebooks/build_02_phase2_training.py
