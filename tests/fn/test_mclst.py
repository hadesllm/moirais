"""Tests for morie.fn.mclst -- Model-based clustering."""

import numpy as np
from morie.fn.mclst import model_based_cluster, mclst
from morie.fn._containers import DescriptiveResult


class TestModelBasedCluster:
    def test_alias(self):
        assert mclst is model_based_cluster

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 2))
        res = model_based_cluster(X, k_range=(2, 4))
        assert isinstance(res, DescriptiveResult)

    def test_optimal_k_in_range(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 2))
        res = model_based_cluster(X, k_range=(2, 5))
        assert 2 <= res.extra["optimal_k"] <= 5

    def test_has_bic_values(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 2))
        res = model_based_cluster(X, k_range=(2, 4))
        assert "bic_values" in res.extra
        assert len(res.extra["bic_values"]) == 3
