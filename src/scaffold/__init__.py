"""
Reasoning scaffold: MAWPS-style word problems, semantic peers, and equation patching.

Heavy dependencies (SymPy, Torch, etc.) load lazily via :func:`__getattr__`.
"""

from __future__ import annotations

from typing import Any, List

__all__: List[str] = ["AgenticSolver", "ReasoningState", "evaluate_predictions"]


def __getattr__(name: str) -> Any:
    if name == "AgenticSolver":
        from scaffold.solver import AgenticSolver

        return AgenticSolver
    if name == "ReasoningState":
        from scaffold.state import ReasoningState

        return ReasoningState
    if name == "evaluate_predictions":
        from scaffold.evaluator import evaluate_predictions

        return evaluate_predictions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
