"""
Agentic solver for MAWPS-style problems: FLAN-T5 equation generation with
semantic-neighbor peer patching (original article reproduction path).

Optional OpenAI generation runs only when explicitly enabled; default matches
the reference pipeline (FLAN-T5 first, then peer scaffold).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from integrations.openai_client import generate_equation_via_openai
from scaffold import prompts
from scaffold import strategies as strat
from scaffold.verifier import is_valid_equation
from utils.semantic_neighbors import NeighborCache


@dataclass
class AgenticSolver:
    """
    Loads FLAN-T5 + sentence-transformers and runs the LLM → peer-patch fallback.

    Parameters
    ----------
    project_root:
        Repository root (contains ``data/raw``).
    device:
        ``\"cuda\"`` or ``\"cpu\"``; inferred when ``None``.
    model_name:
        Hugging Face seq2seq checkpoint (default matches original article code).
    use_openai:
        When True, attempt OpenAI equation generation before FLAN-T5 if a key exists.
    """

    project_root: Path
    device: Optional[str] = None
    model_name: str = "google/flan-t5-base"
    use_openai: bool = False
    _equation_gen: Any = field(default=None, repr=False)
    _embedder: Optional[SentenceTransformer] = field(default=None, repr=False)
    _neighbor_cache: Optional[NeighborCache] = field(default=None, repr=False)
    _train_cache: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _train_embeddings: List[torch.Tensor] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self, train_path: Optional[Path] = None) -> None:
        """Load models and precompute training embeddings for neighbor search."""
        root = Path(self.project_root)
        train_path = train_path or root / "data" / "raw" / "mawps_train.json"

        with train_path.open("r", encoding="utf-8") as f:
            self._train_cache = json.load(f)

        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._neighbor_cache = NeighborCache(shared_model=self._embedder)
        train_problems = [x["problem"] for x in self._train_cache]
        self._train_embeddings = [
            self._embedder.encode(p, convert_to_tensor=True) for p in train_problems
        ]

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        dev_id = 0 if self.device == "cuda" else -1
        self._equation_gen = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=dev_id,
        )

    def _fallback_by_peers(
        self, problem_text: str, k: int = 6
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return patched equation and strategy label, or empty."""
        train_problems = [x["problem"] for x in self._train_cache]
        train_equations = [x["equation"] for x in self._train_cache]
        assert self._neighbor_cache is not None
        top_k = self._neighbor_cache.get_top_k(
            problem_text,
            train_problems,
            self._train_embeddings,
            k=k,
        )
        for idx, _score in top_k:
            peer_eq = train_equations[idx]
            patched_eq, _sol = strat.try_patch_equation(peer_eq, problem_text)
            if patched_eq and is_valid_equation(patched_eq):
                return patched_eq, strat.STRATEGY_PEER_PATCH
        return None, None

    def _generate_flan(self, problem: str) -> str:
        assert self._equation_gen is not None
        prompt = prompts.equation_generation_prompt(problem)
        try:
            result = self._equation_gen(prompt)
            return result[0]["generated_text"].strip()
        except Exception:
            return ""

    def solve_one(self, problem: str) -> Dict[str, Any]:
        """
        Produce ``predicted_equation`` and ``used_strategy`` for a single problem.

        Order: optional OpenAI (if enabled + key) → FLAN-T5 → semantic peer patch.
        """
        used_strategy = strat.STRATEGY_LLM
        equation = ""

        if self.use_openai:
            oa = generate_equation_via_openai(problem)
            if oa and is_valid_equation(oa):
                return {
                    "problem": problem,
                    "predicted_equation": oa,
                    "used_strategy": strat.STRATEGY_OPENAI,
                }

        equation = self._generate_flan(problem)
        if is_valid_equation(equation):
            return {
                "problem": problem,
                "predicted_equation": equation,
                "used_strategy": used_strategy,
            }

        patched_eq, st = self._fallback_by_peers(problem, k=6)
        if patched_eq:
            return {
                "problem": problem,
                "predicted_equation": patched_eq,
                "used_strategy": st or strat.STRATEGY_PEER_PATCH,
            }

        return {
            "problem": problem,
            "predicted_equation": "",
            "used_strategy": strat.STRATEGY_UNSUPPORTED,
        }

    def solve_batch(self, test_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run :meth:`solve_one` for each test row (expects ``problem`` key)."""
        out: List[Dict[str, Any]] = []
        for sample in test_rows:
            out.append(self.solve_one(sample["problem"]))
        return out
