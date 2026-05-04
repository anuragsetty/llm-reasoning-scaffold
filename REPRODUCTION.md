# Reproduction (article-aligned MAWPS pipeline)

This repository reproduces the **MAWPS equation prediction + semantic peer scaffold** workflow described in [Is agentic learning the new machine learning?](https://medium.com/@setgeti/is-agentic-learning-the-new-machine-learning-85732a309dc8) using the code lineage from the original `main_agentic.py` / `evaluate_agentic.py` snapshot.

For repository layout, engineering context, and CI tooling, see the top-level [README.md](README.md). The fastest no-model path is `make setup` then `make run` (validates `data/raw` and evaluates the checked-in baseline predictions).

## Prerequisites

- Python **3.10+** recommended (3.11 tested); CI uses **3.10** on Ubuntu.  
- Disk space for Hugging Face weights (**FLAN-T5-base**, **all-MiniLM-L6-v2**) when running `scripts/run_experiment.py`.  
- No paid API is required for the default path.

## Exact run order

From the repository root (`llm-reasoning-scaffold/`):

1. **Create environment and install**

   ```bash
   make setup
   ```

   Or manually:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

2. **Confirm MAWPS JSON is present**

   ```bash
   python scripts/generate_data.py
   ```

3. **Evaluate the checked-in baseline predictions (no GPU run)**

   This uses `results/metrics/article_baseline_predictions.json` (copied from the parent project’s `data/results.json`) so you can recover **accuracy / correct count** without re-downloading models:

   ```bash
   python scripts/evaluate_results.py
   ```

   Optional: write a small metrics JSON:

   ```bash
   python scripts/evaluate_results.py --metrics-out results/metrics/eval_baseline.json
   ```

4. **Full experiment (regenerates predictions; requires model download)**

   ```bash
   python scripts/run_experiment.py
   python scripts/evaluate_results.py --predictions results/metrics/predictions.json
   ```

   Smoke test on the first 5 problems:

   ```bash
   python scripts/run_experiment.py --limit 5
   python scripts/evaluate_results.py --predictions results/metrics/predictions.json
   ```

5. **Notebook walkthrough**

   Open `notebooks/01_article_reproduction.ipynb` and run top-to-bottom (same steps as above).

## What reproduces the article result

- **Baseline numbers without retraining**: Step **3** evaluates the snapshot file `results/metrics/article_baseline_predictions.json` against `data/raw/mawps_test.json` using SymPy equivalence (same logic as the original `evaluate_agentic.py`).  
- **End-to-end regeneration**: Steps **4** re-run **FLAN-T5 → semantic top‑k neighbors → peer number patch** and write `results/metrics/predictions.json`.

## Note on the legacy layout

The original `main_agentic.py` wrote `agentic_results.json` while `evaluate_agentic.py` read `data/results.json`. This repo **unifies** outputs under `results/metrics/` and documents defaults in this file.
