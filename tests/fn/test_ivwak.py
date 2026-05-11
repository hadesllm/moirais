"""Tests for morie.fn.ivwak — instrumental variable weak test."""
import numpy as np
import pytest
from morie.fn.ivwak import iv_weak_test


class TestIVWeakTest:
    def test_strong_instrument(self):
        rng = np.random.default_rng(42)
        n = 500
        z = rng.standard_normal(n)
        x = 0.8 * z + rng.standard_normal(n) * 0.3
        res = iv_weak_test(x, z)
        assert res.extra["f_statistic"] > 10
        assert res.extra["weak_instrument"] is False

    def test_weak_instrument(self):
        rng = np.random.default_rng(42)
        n = 100
        z = rng.standard_normal(n)
        x = 0.01 * z + rng.standard_normal(n) * 5.0
        res = iv_weak_test(x, z)
        assert res.extra["weak_instrument"] is True

    def test_with_covariates(self):
        rng = np.random.default_rng(42)
        n = 300
        z = rng.standard_normal(n)
        w = rng.standard_normal((n, 2))
        x = 0.7 * z + w @ [0.3, 0.2] + rng.standard_normal(n) * 0.4
        res = iv_weak_test(x, z, covariates=w)
        assert np.isfinite(res.extra["f_statistic"])
