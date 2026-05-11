"""Tests for morie.fn.uncr -- entropy production."""

import numpy as np
from morie.fn.uncr import entropy_production, uncr
from morie.fn._containers import DescriptiveResult


class TestUncr:
    def test_alias(self):
        assert uncr is entropy_production

    def test_uniform_max_entropy(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        r = entropy_production(p, base=2.0)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value - 2.0) < 1e-10

    def test_rate_computation(self):
        p1 = np.array([0.5, 0.5])
        p2 = np.array([0.9, 0.1])
        r = entropy_production(p2, probs_prev=p1, dt=1.0, base=2.0)
        assert "dH_dt" in r.extra
        assert "kl_divergence" in r.extra
