# Rain-robust pothole detection. Run `make help` for targets.
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= .venv/bin/python
endif
ARGS ?=

.PHONY: help env lint download inspect tag convert split rain eda data proposal

help:
	@echo "make env       - create .venv (uv, Python 3.11) and install requirements"
	@echo "make lint      - ruff format + lint"
	@echo "make download  - fetch MWPD, verify+extract RDD2022 (zip downloaded beforehand)"
	@echo "make data      - Phase 1: inspect, convert, split, synthetic rain, EDA"
	@echo "make proposal  - build slides/proposal.pdf and slides/proposal-notes.pdf (speaker notes)"
	@echo "make tag       - rebuild outputs/tag_weather.html (manual MWPD weather tags)"
	@echo "ARGS='--dry-run' or ARGS='--force' are passed to every Phase-1 script"

env:
	uv venv --python 3.11 .venv
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
