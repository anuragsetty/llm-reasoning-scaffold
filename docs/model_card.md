# Model card — LLM reasoning scaffold (MAWPS equation prediction)

## Model details

* **Primary generator:** FLAN-T5–base (local, Hugging Face `transformers`) for equation strings from problem text.
* **Optional generator:** OpenAI Chat Completions when `OPENAI_API_KEY` is set and `--use-openai` is used (`src/integrations/openai_client.py`).
* **Retrieval / structure:** Sentence-Transformer (`all-MiniLM-L6-v2`) for semantic top‑k neighbors; rule-based numeric peer patching when the raw equation fails SymPy validation (`src/scaffold/strategies.py`, `src/utils/semantic_neighbors.py`).
* **Verification:** SymPy parse and lightweight validity checks (`src/scaffold/verifier.py`).

## Intended use

* **Offline research and reproduction** on MAWPS-style single-equation word problems shipped under `data/raw/`.
* **Portfolio / teaching** to show agentic scaffolding (retrieval + repair) around a small LM, not as a production tutor or high-stakes decision system.

## Data

* **Source:** MAWPS-style train/test JSON in `data/raw/` (see `scripts/generate_data.py` for presence checks).
* **Outputs:** Predictions written under `results/metrics/` (e.g. `predictions.json`, checked-in `article_baseline_predictions.json`).

## Metrics

* **Primary:** Equation match rate vs gold using SymPy equivalence (`src/scaffold/evaluator.py`, `scripts/evaluate_results.py`).
* **Diagnostics:** Count of failed parses and mismatches printed or saved as JSON.

## Ethical and safety notes

* Dataset and tasks are **narrow and synthetic/educational** in nature relative to open-domain deployment.
* No demographic or fairness evaluation is performed in this repository; any extension to user-facing products requires separate assessment.

## Limitations

* **Synthetic / benchmark scope:** Results on MAWPS do not generalize to all math word problem benchmarks or free-form reasoning.
* **Baseline stack:** FLAN-T5-base is chosen for accessibility; larger or specialized models would change accuracy profiles.
* **Monitoring:** No production drift detection, online calibration, or content moderation pipeline is included.

## Versioning and reproducibility

* Pin Python (`>=3.10`, CI on 3.10) and lock dependency major versions in `requirements.txt` for your own runs.
* Use `REPRODUCTION.md` and `templates/experiment_log_template.md` to record runs, seeds, and artifact paths when comparing experiments.
