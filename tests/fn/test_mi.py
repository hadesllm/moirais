"""Tests for moirais.fn.mi — mutual information estimation."""
import numpy as np
import pytest
from moirais.fn.mi import mutual_info


class TestMutualInfo:
    def test_correlated(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        y = x + rng.normal(0, 0.1, 500)
        res = mutual_info(x, y)
        assert res.extra["mi_nats"] > 0.1

    def test_independent(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        y = rng.standard_normal(500)
        res = mutual_info(x, y)
        assert res.extra["mi_nats"] < 0.3

    def test_nbins(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        y = x * 2 + rng.normal(0, 0.5, 200)
        res10 = mutual_info(x, y, n_bins=10)
        res30 = mutual_info(x, y, n_bins=30)
        assert res10.extra["mi_nats"] > 0
        assert res30.extra["mi_nats"] > 0
