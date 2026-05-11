"""Tests for morie.fn.kcvms — kernel Cramer-von Mises test."""

import numpy as np
import pytest

from morie.fn.kcvms import kcvms


class TestKcvms:
    def test_normal_not_rejected(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = kcvms(data, cdf_func="normal", n_boot=199, seed=42)
        assert res["p_value"] > 0.01

    def test_statistic_positive(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 50)
        res = kcvms(data, n_boot=99, seed=42)
        assert res["statistic"] >= 0

    def test_uniform_fit(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, 100)
        res = kcvms(data, cdf_func="uniform", n_boot=99, seed=42)
        assert res["p_value"] > 0.01

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kcvms(np.array([1.0, 2.0]))
