"""Tests for morie.fn.binom -- Exact binomial test."""

import pytest
from morie.fn.binom import binomial_test
from morie.fn._containers import TestResult


class TestBinomial:
    def test_fair_coin(self):
        """50 heads in 100 flips => p ~ 1 for H0: p=0.5."""
        r = binomial_test(50, 100, 0.5)
        assert isinstance(r, TestResult)
        assert r.p_value > 0.9

    def test_biased_coin(self):
        """90 heads in 100 flips => reject H0: p=0.5."""
        r = binomial_test(90, 100, 0.5)
        assert r.p_value < 0.001

    def test_one_sided_greater(self):
        """One-sided test for proportion > 0.5."""
        r = binomial_test(60, 100, 0.5, alternative="greater")
        assert r.p_value < 0.05

    def test_raises_bad_x(self):
        with pytest.raises(ValueError):
            binomial_test(101, 100, 0.5)

    def test_raises_bad_p0(self):
        with pytest.raises(ValueError):
            binomial_test(5, 10, 1.5)
