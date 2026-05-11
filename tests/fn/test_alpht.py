"""Tests for morie.fn.alpht -- Bayesian prior elicitation."""

from morie.fn.alpht import prior_elicit, alpht
from morie.fn._containers import DescriptiveResult


class TestAlpht:
    def test_alias(self):
        assert alpht is prior_elicit

    def test_normal(self):
        quantiles = {0.05: 10, 0.50: 50, 0.95: 90}
        r = prior_elicit(quantiles, family="normal")
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value["mu"] - 50) < 5

    def test_beta(self):
        quantiles = {0.1: 0.2, 0.5: 0.5, 0.9: 0.8}
        r = prior_elicit(quantiles, family="beta")
        assert r.value["alpha"] > 0
        assert r.value["beta"] > 0
