"""Tests for morie.fn.kscvs — kernel-smoothed KS test."""

import numpy as np
import pytest

from morie.fn.kscvs import kscvs


class TestKscvs:
    def test_normal_not_rejected(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = kscvs(data, cdf_func="normal", n_boot=199, seed=42)
        assert res["p_value"] > 0.01

    def test_nonnormal_rejected(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(1.0, 200)
        res = kscvs(data, cdf_func="normal", n_boot=199, seed=42)
        assert res["p_value"] < 0.1

    def test_statistic_positive(self):
        data = np.arange(1, 51, dtype=float)
        res = kscvs(data, n_boot=99, seed=42)
        assert res["statistic"] > 0

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kscvs(np.array([1.0, 2.0, 3.0]))

    def test_raises_unknown_cdf(self):
        with pytest.raises(ValueError):
            kscvs(np.arange(10, dtype=float), cdf_func="bogus")
