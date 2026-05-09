"""Tests for moirais.fn.elbmk -- Elbow method for k."""

import numpy as np
from moirais.fn.elbmk import elbow_method, elbmk
from moirais.fn._containers import DescriptiveResult


class TestElbowMethod:
    def test_alias(self):
        assert elbmk is elbow_method

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 3))
        res = elbow_method(X, k_range=(2, 5))
        assert isinstance(res, DescriptiveResult)

    def test_optimal_k_in_range(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 3))
        res = elbow_method(X, k_range=(2, 6))
        assert 2 <= res.value <= 6

    def test_inertias_decreasing(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 3))
        res = elbow_method(X, k_range=(2, 5))
        inertias = res.extra["inertias"]
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i+1] - 1e-6
