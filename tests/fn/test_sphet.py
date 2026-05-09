"""Tests for moirais.fn.sphet — spatial heterogeneity."""
import numpy as np
import pytest
from moirais.fn.sphet import spatial_heterogeneity


class TestSpatialHeterogeneity:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 30
        W = np.eye(n, k=1) + np.eye(n, k=-1)
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1)
        residuals = rng.standard_normal(n)
        res = spatial_heterogeneity(residuals, W)
        assert 0 <= res.p_value <= 1

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            spatial_heterogeneity(np.ones(5), np.eye(3))
