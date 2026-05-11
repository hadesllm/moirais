"""Tests for morie.fn.wdvar — Windowed variogram cloud."""

import numpy as np
import pytest

from morie.fn.wdvar import wdvar


class TestWdvar:

    def test_output_keys(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = wdvar(coords, values, n_bins=10)
        assert len(result["lags"]) == 10
        assert len(result["counts"]) == 10

    def test_cloud_non_empty(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        values = rng.standard_normal(15)
        result = wdvar(coords, values)
        assert len(result["cloud_dists"]) > 0
        assert len(result["cloud_gamma"]) == len(result["cloud_dists"])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            wdvar(np.ones((5, 2)), np.ones(3))
