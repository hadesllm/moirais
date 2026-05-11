"""Tests for morie.fn.bhdi — Highest density interval."""

import numpy as np
import pytest

from morie.fn.bhdi import bayesian_hdi


class TestBayesianHDI:
    def test_normal(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 10000)
        res = bayesian_hdi(samples, prob=0.95)
        assert res.ci_lower < 0 < res.ci_upper
        assert res.estimate < 4.5

    def test_narrow_prob(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 10000)
        r50 = bayesian_hdi(samples, prob=0.50)
        r95 = bayesian_hdi(samples, prob=0.95)
        assert r50.estimate < r95.estimate

    def test_too_few(self):
        with pytest.raises(ValueError):
            bayesian_hdi([1, 2])
