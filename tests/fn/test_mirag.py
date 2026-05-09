"""Tests for moirais.fn.mirag -- differential privacy Laplace mechanism."""

import numpy as np
from moirais.fn.mirag import dp_laplace, mirag
from moirais.fn._containers import DescriptiveResult


class TestMirag:
    def test_alias(self):
        assert mirag is dp_laplace

    def test_mean_query(self):
        x = np.arange(100, dtype=float)
        r = dp_laplace(x, epsilon=10.0, query="mean")
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value - r.extra["true_value"]) < 5

    def test_high_privacy(self):
        x = np.arange(1, 101, dtype=float)
        r1 = dp_laplace(x, epsilon=0.01)
        r2 = dp_laplace(x, epsilon=100.0)
        assert r1.extra["scale"] > r2.extra["scale"]
