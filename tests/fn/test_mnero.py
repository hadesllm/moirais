"""Tests for moirais.fn.mnero -- cellular automaton."""

from moirais.fn.mnero import cellular_automaton, mnero
from moirais.fn._containers import DescriptiveResult


class TestMnero:
    def test_alias(self):
        assert mnero is cellular_automaton

    def test_game_of_life(self):
        result = cellular_automaton(size=20, n_steps=50, rule="life", seed=42)
        assert isinstance(result, DescriptiveResult)
        assert 0 <= result.value <= 1

    def test_urban(self):
        result = cellular_automaton(size=15, n_steps=30, rule="urban", seed=42, density=0.1)
        assert result.extra["rule"] == "urban"
        assert len(result.extra["density_history"]) > 0
