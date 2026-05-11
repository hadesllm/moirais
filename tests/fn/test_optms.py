"""Tests for morie.fn.optms -- Wasserstein optimal transport."""

import numpy as np
from morie.fn.optms import optimal_transport, optms
from morie.fn._containers import DescriptiveResult


class TestOptms:
    def test_alias(self):
        assert optms is optimal_transport

    def test_identical_distributions(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = optimal_transport(x, x)
        assert isinstance(r, DescriptiveResult)
        assert r.value == 0.0

    def test_shifted_distributions(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(3, 1, 500)
        r = optimal_transport(x, y)
        assert 2.5 < r.value < 3.5
