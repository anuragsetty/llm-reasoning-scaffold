"""Prompt templates for equation generation (matches original one-shot FLAN-T5 input)."""


def equation_generation_prompt(problem: str) -> str:
    """
    Build the text2text prompt for equation generation.

    The reference implementation passed the raw problem string through FLAN-T5;
    this function preserves that behavior while centralizing future edits.
    """
    return problem.strip()
