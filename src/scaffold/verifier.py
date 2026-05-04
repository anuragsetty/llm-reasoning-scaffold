"""Equation validity checks (SymPy parseability)."""

from __future__ import annotations

from sympy import sympify
from sympy.core.sympify import SympifyError


def is_valid_equation(eq: str) -> bool:
    """Return True if ``eq`` can be parsed as a SymPy expression."""
    if not eq or not str(eq).strip():
        return False
    try:
        sympify(eq)
        return True
    except (SympifyError, TypeError, ValueError):
        return False
