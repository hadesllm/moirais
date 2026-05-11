"""Tests for morie.fn.herit -- GREML heritability."""

import numpy as np
import pytest
from morie.fn.herit import herit


def _make_data(h2_true=0.5, n=50, p=100, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.choice([0, 1, 2], size=(n, p)).astype(float)
    pf = np.mean(Z, axis=0) / 2.0
    pf = np.clip(pf, 0.01, 0.99)
    M = Z - 2.0 * pf
    G = (M @ M.T) / (2.0 * np.sum(pf * (1 - pf)))
    L = np.linalg.cholesky(G + np.eye(n) * 0.01)
    g = L @ rng.standard_normal(n) * np.sqrt(h2_true)
    e = rng.standard_normal(n) * np.sqrt(1 - h2_true)
    y = g + e
    return y, G


class TestHerit:
    def test_h2_bounded(self):
        y, G = _make_data()
        res = herit(y, G, max_iter=20)
        assert 0 <= res.statistic <= 1

    def test_var_components_positive(self):
        y, G = _make_data()
        res = herit(y, G, max_iter=20)
        assert res.extra["var_g"] > 0
        assert res.extra["var_e"] > 0

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            herit(np.ones(10), np.eye(5))

    def test_returns_name(self):
        y, G = _make_data()
        res = herit(y, G, max_iter=10)
        assert res.name == "GREML_h2"
