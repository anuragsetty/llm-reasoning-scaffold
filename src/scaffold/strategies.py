"""
Strategy identifiers and peer-based equation patching (numeric alignment).

The peer scaffold adapts a neighbor's equation template to the current problem
by substituting numbers in template order, then validates with SymPy.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from sympy import Symbol, solve, sympify
from sympy.core.sympify import SympifyError

STRATEGY_LLM = "llm"
STRATEGY_PEER_PATCH = "peer_patch"
STRATEGY_UNSUPPORTED = "unsupported"
STRATEGY_OPENAI = "openai"


def extract_numbers(text: str) -> List[int]:
    """Extract signed integers from ``text`` in left-to-right order."""
    return [int(s) for s in re.findall(r"[-+]?\d+", text)]


def generalize_equation(template_eq: str, test_text: str) -> str:
    """Replace template numbers with numbers from ``test_text`` in order."""
    test_numbers = extract_numbers(test_text)
    template_numbers = extract_numbers(template_eq)
    patched_eq = template_eq
    for old, new in zip(template_numbers, test_numbers):
        patched_eq = patched_eq.replace(str(old), str(new), 1)
    return patched_eq


def try_patch_equation(template_eq: str, test_text: str) -> Tuple[Optional[str], Optional[list]]:
    """
    Patch ``template_eq`` using numbers from ``test_text`` and attempt to solve for ``x``.

    Returns ``(new_equation, solutions)`` or ``(None, None)`` on failure.
    """
    try:
        new_eq = generalize_equation(template_eq, test_text)
        expr = sympify(new_eq)
        x = Symbol("x")
        sol = solve(expr, x)
        return new_eq, sol
    except (SympifyError, TypeError, ValueError, Exception):
        return None, None
