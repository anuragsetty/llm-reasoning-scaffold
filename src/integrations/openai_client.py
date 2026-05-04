"""
Optional OpenAI client for equation generation.

If ``OPENAI_API_KEY`` is unset or the SDK call fails, helpers return ``None`` so
callers can fall back to local FLAN-T5 and rule-based peer patching.
"""

from __future__ import annotations

import os
from typing import Optional


def is_openai_configured() -> bool:
    """Return True when an API key is present in the environment."""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def generate_equation_via_openai(problem: str, model: str = "gpt-4o-mini") -> Optional[str]:
    """
    Ask OpenAI for a single-line symbolic equation (SymPy-parseable).

    Returns ``None`` if not configured, on import errors, or on API failure.
    """
    if not is_openai_configured():
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI()
    system = (
        "You output only a single arithmetic equation for the word problem using "
        "placeholders N_00, N_01, ... exactly as in the problem. No words, no JSON."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": problem},
            ],
            temperature=0,
            max_tokens=128,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None
