"""Tests for moirais.fn.bbin — Bayesian binomial."""

import pytest

from moirais.fn.bbin import bayesian_binomial


class TestBayesianBinomial:
    def test_uniform_prior(self):
        res = bayesian_binomial(30, 100)
        assert 0.2 < res.value < 0.4

    def test_ci_contains_mean(self):
        res = bayesian_binomial(50, 100)
        assert res.extra["ci_lower"] < res.value < res.extra["ci_upper"]

    def test_strong_prior(self):
        res = bayesian_binomial(5, 10, prior_a=100, prior_b=100)
        assert abs(res.value - 0.5) < 0.1

    def test_invalid(self):
        with pytest.raises(ValueError):
            bayesian_binomial(20, 10)
