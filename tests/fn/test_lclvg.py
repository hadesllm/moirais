"""Tests for morie.fn.lclvg — Local variogram estimation."""

import numpy as np
import pytest

from morie.fn.lclvg import lclvg


class TestLclvg:

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = lclvg(coords, values, n_lags=8)
        assert result["semivariance"].shape[1] == 8
        assert result["n_targets"] == 20

    def test_custom_target(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        values = rng.standard_normal(15)
        target = np.array([[5.0, 5.0]])
        result = lclvg(coords, values, target=target)
        assert result["n_targets"] == 1

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            lclvg(np.ones((5, 2)), np.ones(3))
