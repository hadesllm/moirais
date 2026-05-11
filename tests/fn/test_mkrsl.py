"""Tests for morie.fn.mkrsl -- Marker selection."""

import numpy as np
import pytest
from morie.fn.mkrsl import mkrsl


class TestMkrsl:
    def test_marginal_method(self):
        rng = np.random.default_rng(42)
        n, p = 50, 20
        Z = rng.choice([0, 1, 2], size=(n, p)).astype(float)
        beta = np.zeros(p)
        beta[0] = 2.0
        y = Z @ beta + rng.standard_normal(n) * 0.5
        res = mkrsl(y, Z, method="marginal")
        assert 0 in res.extra["selected_indices"]

    def test_ridge_method(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(30, 10)).astype(float)
        y = Z[:, 0] * 2.0 + rng.standard_normal(30) * 0.5
        res = mkrsl(y, Z, method="ridge_importance")
        assert res.statistic > 0

    def test_correlation_method(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(30, 10)).astype(float)
        y = rng.standard_normal(30)
        res = mkrsl(y, Z, method="correlation")
        assert res.statistic > 0

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            mkrsl(np.ones(10), np.ones((10, 5)), method="unknown")

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            mkrsl(np.ones(5), np.ones((10, 5)))
