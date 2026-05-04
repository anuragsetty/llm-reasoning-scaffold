"""
Linear reasoning graph skeleton aligned with the agentic scaffold stages.

Nodes map to human-interpretable phases; ``solve_stepwise`` delegates to
:class:`scaffold.solver.AgenticSolver` so behavior stays consistent with scripts.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangGraph is required for graphs.reasoning_graph. "
        "Use Python 3.10+ and `pip install -r requirements.txt`."
    ) from exc

from scaffold.solver import AgenticSolver
from scaffold.state import ReasoningState
from scaffold.verifier import is_valid_equation


def _understand_problem(state: ReasoningState) -> Dict[str, Any]:
    """Capture a short working summary of the raw problem text."""
    problem = state.get("problem", "")
    summary = problem.strip()[:500]
    return {"understood_summary": summary}


def _select_strategy(state: ReasoningState) -> Dict[str, Any]:
    """Declare the high-level plan (local LLM then semantic peer scaffold)."""
    return {"strategy": "flan_t5_then_semantic_peer_patch"}


def _make_solve_stepwise(solver: AgenticSolver) -> Callable[[ReasoningState], Dict[str, Any]]:
    def _solve_stepwise(state: ReasoningState) -> Dict[str, Any]:
        problem = state["problem"]
        result = solver.solve_one(problem)
        return {
            "equation": result.get("predicted_equation", ""),
            "used_strategy": result.get("used_strategy", ""),
        }

    return _solve_stepwise


def _verify_answer(state: ReasoningState) -> Dict[str, Any]:
    """SymPy-parse check on the predicted equation (empty string fails)."""
    eq = (state.get("equation") or "").strip()
    ok = bool(eq) and is_valid_equation(eq)
    msg = "equation parses" if ok else "invalid or empty equation"
    return {"verification_ok": ok, "verification_message": msg}


def _summarize_result(state: ReasoningState) -> Dict[str, Any]:
    """Produce a one-line human-readable recap for logging or UI."""
    parts = [
        f"strategy={state.get('used_strategy', '')}",
        f"ok={state.get('verification_ok', False)}",
        f"eq={state.get('equation', '')}",
    ]
    return {"summary": "; ".join(parts)}


def build_reasoning_graph(solver: AgenticSolver) -> Any:
    """
    Build and compile a LangGraph workflow over :class:`~scaffold.state.ReasoningState`.

    Parameters
    ----------
    solver:
        A loaded :class:`~scaffold.solver.AgenticSolver` instance (call ``load()`` first).

    Returns
    -------
    CompiledGraph
        Runnable LangGraph artifact (``.invoke()`` / ``.stream()``).
    """
    graph = StateGraph(ReasoningState)
    graph.add_node("understand_problem", _understand_problem)
    graph.add_node("select_strategy", _select_strategy)
    graph.add_node("solve_stepwise", _make_solve_stepwise(solver))
    graph.add_node("verify_answer", _verify_answer)
    graph.add_node("summarize_result", _summarize_result)

    graph.add_edge(START, "understand_problem")
    graph.add_edge("understand_problem", "select_strategy")
    graph.add_edge("select_strategy", "solve_stepwise")
    graph.add_edge("solve_stepwise", "verify_answer")
    graph.add_edge("verify_answer", "summarize_result")
    graph.add_edge("summarize_result", END)

    return graph.compile()
