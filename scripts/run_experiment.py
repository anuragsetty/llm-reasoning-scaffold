#!/usr/bin/env python3
"""
Run the agentic MAWPS experiment (FLAN-T5 + semantic peer patch).

Writes ``results/metrics/predictions.json`` compatible with ``evaluate_results.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from scaffold.solver import AgenticSolver  # noqa: E402
from utils.io import load_json, repo_root, save_json  # noqa: E402
from utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAWPS agentic solver.")
    parser.add_argument("--root", type=Path, default=repo_root(), help="Repo root.")
    parser.add_argument(
        "--test-path",
        type=Path,
        default=None,
        help="Override test JSON path (default: data/raw/mawps_test.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only solve the first N problems (for smoke tests).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output predictions JSON (default: results/metrics/predictions.json).",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="If OPENAI_API_KEY is set, try OpenAI before FLAN-T5 (off by default).",
    )
    args = parser.parse_args()

    root: Path = args.root
    test_path = args.test_path or root / "data" / "raw" / "mawps_test.json"
    out_path = args.out or root / "results" / "metrics" / "predictions.json"

    test_rows = load_json(test_path)
    if args.limit is not None:
        test_rows = test_rows[: args.limit]

    log.info("Loading models (downloads on first run)...")
    solver = AgenticSolver(project_root=root, use_openai=args.use_openai)
    solver.load()

    log.info("Solving %d problems...", len(test_rows))
    results = solver.solve_batch(test_rows)
    save_json(out_path, results)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
