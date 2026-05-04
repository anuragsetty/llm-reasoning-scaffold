"""Semantic similarity over problem text (sentence-transformers)."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import torch
from sentence_transformers import SentenceTransformer, util


class NeighborCache:
    """
    Encode queries and rank corpus items by cosine similarity.

    Optionally reuses a shared :class:`SentenceTransformer` to avoid loading
    the embedder twice (train query encoder + neighbor search).
    """

    def __init__(
        self,
        embed_model: str = "all-MiniLM-L6-v2",
        cache_file: Optional[str] = None,
        shared_model: Optional[SentenceTransformer] = None,
    ) -> None:
        self.model = shared_model or SentenceTransformer(embed_model)
        self.cache_file = cache_file
        self.cache: dict = {}
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def encode(self, text: str) -> torch.Tensor:
        """Return a single embedding tensor for ``text``."""
        return self.model.encode(text, convert_to_tensor=True)

    def save(self) -> None:
        """Persist cache to disk if ``cache_file`` was provided."""
        if self.cache_file:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)

    def get_top_k(
        self,
        query_text: str,
        corpus_texts: Sequence[str],
        corpus_embeddings: Optional[Sequence[torch.Tensor]] = None,
        k: int = 4,
    ) -> List[Tuple[int, float]]:
        """Return ``k`` (index, score) pairs sorted by similarity."""
        query_embedding = self.encode(query_text)
        if corpus_embeddings is None:
            corpus_embeddings = [self.encode(text) for text in corpus_texts]
        scores = util.pytorch_cos_sim(query_embedding, torch.stack(list(corpus_embeddings)))[0]
        top_results = torch.topk(scores, k=min(k, scores.shape[0]))
        return [(int(i), float(scores[int(i)])) for i in top_results.indices]
