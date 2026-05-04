.PHONY: setup run check lint format

PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

# Fast path: validate data files and evaluate checked-in baseline predictions (no model download).
run:
	$(PY) scripts/generate_data.py
	$(PY) scripts/evaluate_results.py

check:
	$(PY) -m pytest -q

format:
	$(PY) -m black src tests scripts

lint:
	$(PY) -m ruff check src tests scripts
