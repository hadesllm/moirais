"""Tests for moirais.fn.ktwos — kernel two-sample test (MMD)."""

import numpy as np
import pytest

from moirais.fn.ktwos import ktwos


class TestKtwos:
    def test_same_distribution_not_rejected(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        res = ktwos(x, y, n_perm=199, seed=42)
        assert res["p_value"] > 0.01

    def test_different_distributions_rejected(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(3, 1, 100)
        res = ktwos(x, y, n_perm=199, seed=42)
        assert res["p_value"] < 0.05

    def test_mmd2_nonneg_same(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        res = ktwos(x, y, n_perm=99, seed=42)
        assert abs(res["mmd2"]) < 0.5

    def test_raises_small(self):
        with pytest.raises(ValueError):
            ktwos(np.array([1.0]), np.array([1.0, 2.0]))
