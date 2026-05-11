"""Tests for morie.fn.gblup -- Genomic BLUP."""

import numpy as np
import pytest
from morie.fn.gblup import gblup


def _make_data(n=50, p=20, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.choice([0, 1, 2], size=(n, p))
    pf = np.mean(Z, axis=0) / 2.0
    pf = np.clip(pf, 0.01, 0.99)
    M = Z - 2.0 * pf
    G = (M @ M.T) / (2.0 * np.sum(pf * (1 - pf)))
    true_g = rng.standard_normal(n) * 0.5
    y = 5.0 + G @ true_g + rng.standard_normal(n) * 0.3
    X = np.ones((n, 1))
    return y, X, G


class TestGblup:
    def test_returns_genomics_result(self):
        y, X, G = _make_data()
        res = gblup(y, X, G)
        assert res.name == "GBLUP"

    def test_gebv_length(self):
        y, X, G = _make_data()
        res = gblup(y, X, G)
        assert len(res.extra["gebv"]) == len(y)

    def test_beta_length(self):
        y, X, G = _make_data()
        res = gblup(y, X, G)
        assert len(res.extra["beta"]) == X.shape[1]

    def test_statistic_positive(self):
        y, X, G = _make_data()
        res = gblup(y, X, G)
        assert res.statistic >= 0

    def test_dimension_mismatch(self):
        y, X, G = _make_data()
        with pytest.raises(ValueError):
            gblup(y[:10], X, G)
