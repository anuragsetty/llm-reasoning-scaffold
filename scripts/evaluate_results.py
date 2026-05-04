#!/usr/bin/env python3
"""
Compute equation-match accuracy against MAWPS gold equations (SymPy equivalence).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from scaffold.evaluator import evaluate_predictions  # noqa: E402
from utils.io import load_json, repo_root, save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction JSON vs MAWPS gold.")
    parser.add_argument("--root", type=Path, default=repo_root(), help="Repo root.")
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Gold test JSON (default: data/raw/mawps_test.json).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Predictions JSON. Default: results/metrics/article_baseline_predictions.json "
            "(checked in snapshot for article-aligned numbers)."
        ),
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional path to write metrics JSON.",
    )
    args = parser.parse_args()

    root: Path = args.root
    gold_path = args.gold or root / "data" / "raw" / "mawps_test.json"
    pred_path = (
        args.predictions or root / "results" / "metrics" / "article_baseline_predictions.json"
    )

    gold_rows = load_json(gold_path)
    preds = load_json(pred_path)

    correct, total, parse_fail, accuracy = evaluate_predictions(gold_rows, preds)
    print(f"Correct: {correct} / {total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Failed parses: {parse_fail}")

    if args.metrics_out:
        save_json(
            args.metrics_out,
            {
                "correct": correct,
                "total": total,
                "parse_fail": parse_fail,
                "accuracy": accuracy,
                "gold_path": str(gold_path),
                "predictions_path": str(pred_path),
            },
        )


if __name__ == "__main__":
    main()
