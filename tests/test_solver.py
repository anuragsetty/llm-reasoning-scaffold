"""Lightweight tests for peer patching (no Hugging Face downloads)."""

from scaffold import strategies as strat
from scaffold.verifier import is_valid_equation


def test_peer_patch_substitutes_numbers():
    template = "N_00 + N_01"
    text = "Ann has 3 apples and 5 oranges."
    patched, _sol = strat.try_patch_equation(template, text)
    assert patched is not None
    assert "3" in patched and "5" in patched


def test_try_patch_equation_invalid_template():
    patched, sol = strat.try_patch_equation("+++", "1 2 3")
    assert patched is None and sol is None


def test_is_valid_equation_after_patch():
    peer = "N_00 - N_01"
    problem = "Norma has 10 cards . She loses 3 . How many cards will Norma have ?"
    patched, _ = strat.try_patch_equation(peer, problem)
    assert patched is not None
    assert is_valid_equation(patched)
