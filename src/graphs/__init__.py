"""LangGraph workflow definitions (lazy import so LangGraph stays optional at import time)."""

from __future__ import annotations

from typing import Any, List

__all__: List[str] = ["build_reasoning_graph"]


def __getattr__(name: str) -> Any:
    if name == "build_reasoning_graph":
        from graphs.reasoning_graph import build_reasoning_graph

        return build_reasoning_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
