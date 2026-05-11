"""Tests for morie.fn.gambt -- card probability."""

from morie.fn.gambt import card_probability, gambt
from morie.fn._containers import DescriptiveResult


class TestGambt:
    def test_alias(self):
        assert gambt is card_probability

    def test_four_aces(self):
        r = card_probability(n_total=52, n_target=4, draw=5, at_least=1)
        assert isinstance(r, DescriptiveResult)
        assert 0 < r.value["probability"] < 1
        assert abs(sum(r.value["pmf"].values()) - 1.0) < 1e-10

    def test_certain_draw(self):
        r = card_probability(n_total=4, n_target=4, draw=4, at_least=4)
        assert abs(r.value["probability"] - 1.0) < 1e-10
