"""Tests for moirais.fn.lrp — Likelihood ratios."""

import pytest

from moirais.fn.lrp import likelihood_ratios


class TestLikelihoodRatios:
    def test_good_test(self):
        res = likelihood_ratios(0.95, 0.90)
        assert res.extra["lr_positive"] == pytest.approx(9.5)
        assert res.extra["lr_negative"] == pytest.approx(0.05 / 0.9)

    def test_perfect_specificity(self):
        res = likelihood_ratios(0.8, 1.0)
        assert res.extra["lr_positive"] == float("inf")

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            likelihood_ratios(1.5, 0.9)
