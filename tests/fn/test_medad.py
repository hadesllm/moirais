"""Tests for moirais.fn.medad — mediation analysis."""
import numpy as np
import pytest
from moirais.fn.medad import mediation_analysis


class TestMediationAnalysis:
    def test_basic_mediation(self):
        rng = np.random.default_rng(42)
        n = 200
        treatment = rng.choice([0, 1], size=n)
        mediator = 0.5 * treatment + rng.standard_normal(n) * 0.3
        outcome = 0.4 * mediator + 0.3 * treatment + rng.standard_normal(n) * 0.3
        res = mediation_analysis(outcome, treatment, mediator)
        assert np.isfinite(res.extra["indirect_effect"])
        assert np.isfinite(res.extra["direct_effect"])
        assert np.isfinite(res.extra["total_effect"])

    def test_total_equals_sum(self):
        rng = np.random.default_rng(42)
        n = 300
        t = rng.choice([0, 1], size=n)
        m = 0.6 * t + rng.standard_normal(n) * 0.2
        y = 0.5 * m + 0.2 * t + rng.standard_normal(n) * 0.2
        res = mediation_analysis(y, t, m)
        total = res.extra["total_effect"]
        parts = res.extra["indirect_effect"] + res.extra["direct_effect"]
        assert abs(total - parts) < 1e-10

    def test_n_matches(self):
        rng = np.random.default_rng(42)
        n = 150
        t = rng.choice([0, 1], size=n)
        m = rng.standard_normal(n)
        y = rng.standard_normal(n)
        res = mediation_analysis(y, t, m)
        assert res.extra["n"] == n
