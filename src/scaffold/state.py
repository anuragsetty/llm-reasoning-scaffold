"""
LangGraph-compatible shared state for a single word-problem solving episode.

Fields are optional where noted so nodes can incrementally populate the state.
"""

from __future__ import annotations

from typing import List, NotRequired, TypedDict


class ReasoningState(TypedDict, total=False):
    """State carried through the reasoning graph."""

    problem: str
    understood_summary: str
    strategy: str
    equation: str
    used_strategy: str
    verification_ok: bool
    verification_message: str
    summary: str
    errors: List[str]
    openai_attempted: NotRequired[bool]
