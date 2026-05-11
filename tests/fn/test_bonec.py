"""Tests for morie.fn.bonec -- Weibull failure analysis."""

import numpy as np
from morie.fn.bonec import weibull_analysis, bonec
from morie.fn._containers import DescriptiveResult


class TestBonec:
    def test_alias(self):
        assert bonec is weibull_analysis

    def test_basic_fit(self):
        rng = np.random.default_rng(42)
        from scipy import stats
        t = stats.weibull_min.rvs(2.0, scale=10.0, size=100, random_state=rng)
        r = weibull_analysis(t)
        assert isinstance(r, DescriptiveResult)
        assert 1.0 < r.value["shape"] < 4.0
        assert 5.0 < r.value["scale"] < 20.0

    def test_mttf_positive(self):
        t = np.array([5, 10, 15, 20, 25, 30], dtype=float)
        r = weibull_analysis(t)
        assert r.value["mttf"] > 0
