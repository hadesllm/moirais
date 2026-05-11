"""Tests for morie.fn.mtrpr -- Multi-trait genomic prediction."""

import numpy as np
import pytest
from morie.fn.mtrpr import mtrpr


def _make_data(n=30, p=20, t=2, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.choice([0, 1, 2], size=(n, p)).astype(float)
    pf = np.mean(Z, axis=0) / 2.0
    pf = np.clip(pf, 0.01, 0.99)
    M = Z - 2.0 * pf
    G = (M @ M.T) / (2.0 * np.sum(pf * (1 - pf)))
    Y = np.column_stack([
        G @ rng.standard_normal(n) * 0.5 + rng.standard_normal(n) * 0.3
        for _ in range(t)
    ])
    return Y, G


class TestMtrpr:
    def test_gebv_shape(self):
        Y, G = _make_data()
        res = mtrpr(Y, G)
        gebv = np.array(res.extra["gebv"])
        assert gebv.shape == Y.shape

    def test_accuracy_per_trait(self):
        Y, G = _make_data()
        res = mtrpr(Y, G)
        acc = res.extra["accuracy_per_trait"]
        assert len(acc) == 2

    def test_single_trait(self):
        Y, G = _make_data(t=1)
        res = mtrpr(Y, G)
        assert res.extra["n_traits"] == 1

    def test_dimension_mismatch(self):
        Y, G = _make_data()
        with pytest.raises(ValueError):
            mtrpr(Y[:10], G)
