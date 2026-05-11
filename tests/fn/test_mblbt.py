"""Tests for morie.fn.mblbt -- Moving block bootstrap."""

import numpy as np
import pytest
from morie.fn.mblbt import moving_block_bootstrap


class TestMovingBlockBootstrap:
    def test_basic_mean(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        r = moving_block_bootstrap(x)
        assert abs(r["estimate"] - np.mean(x)) < 1e-10
        assert r["se"] > 0
        assert r["ci_lower"] < r["ci_upper"]

    def test_explicit_block_size(self):
        x = np.arange(20, dtype=float)
        r = moving_block_bootstrap(x, block_size=5, n_boot=100)
        assert r["block_size"] == 5

    def test_median_statistic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        r = moving_block_bootstrap(x, statistic="median")
        assert r["statistic"] == "median"

    def test_boot_distribution_length(self):
        x = np.arange(30, dtype=float)
        r = moving_block_bootstrap(x, n_boot=200)
        assert len(r["boot_distribution"]) == 200

    def test_too_short(self):
        with pytest.raises(ValueError, match="at least 4"):
            moving_block_bootstrap(np.array([1.0, 2.0, 3.0]))

    def test_invalid_statistic(self):
        with pytest.raises(ValueError, match="statistic must be"):
            moving_block_bootstrap(np.arange(10, dtype=float), statistic="bad")
