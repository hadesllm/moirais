"""Tests for morie.fn.kdgof — kernel goodness-of-fit test."""

import numpy as np
import pytest

from morie.fn.kdgof import kdgof


class TestKdgof:
    def test_normal_not_rejected(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = kdgof(data, cdf_func="normal", n_boot=199, seed=42)
        assert res["p_value"] > 0.01

    def test_nonnormal_detected(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(1.0, 200)
        res = kdgof(data, cdf_func="normal", n_boot=199, seed=42)
        assert res["p_value"] < 0.15

    def test_statistic_nonneg(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 50)
        res = kdgof(data, n_boot=49, seed=42)
        assert res["statistic"] >= 0

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kdgof(np.array([1.0, 2.0]))
