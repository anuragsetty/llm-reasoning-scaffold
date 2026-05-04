"""
Accuracy-style evaluation: compare predicted equations to gold on MAWPS items.

Uses SymPy ``simplify(parse_expr(gold) - parse_expr(pred)) == 0`` equivalence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sympy import parse_expr, simplify
from sympy.core.sympify import SympifyError


def evaluate_predictions(
    gold_rows: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> Tuple[int, int, int, float]:
    """
    Evaluate structured predictions against gold equations keyed by ``problem``.

    Each prediction dict should include ``problem`` and ``predicted_equation``.

    Returns ``(correct, total, parse_fail, accuracy)``.
    """
    gold_lookup = {item["problem"]: item["equation"] for item in gold_rows}
    correct = 0
    total = 0
    parse_fail = 0

    for r in predictions:
        total += 1
        try:
            gold_eq = gold_lookup[r["problem"]]
            pred_eq = r["predicted_equation"]
            if simplify(parse_expr(gold_eq) - parse_expr(pred_eq)) == 0:
                correct += 1
        except (SympifyError, KeyError, Exception):
            parse_fail += 1

    accuracy = (correct / total) if total else 0.0
    return correct, total, parse_fail, accuracy
