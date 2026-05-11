"""Tests for morie.fn.blkrg — Block kriging."""

import numpy as np
import pytest

from morie.fn.blkrg import blkrg


class TestBlkrg:

    def test_prediction_shape(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        blocks = np.array([[3.0, 3.0], [7.0, 7.0]])
        result = blkrg(coords, values, blocks, seed=42)
        assert result["predictions"].shape == (2,)
        assert result["n_blocks"] == 2

    def test_variances_non_negative(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        values = rng.standard_normal(15)
        blocks = np.array([[5.0, 5.0]])
        result = blkrg(coords, values, blocks, seed=42)
        assert np.all(result["variances"] >= 0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            blkrg(np.ones((5, 2)), np.ones(3), np.ones((1, 2)))
