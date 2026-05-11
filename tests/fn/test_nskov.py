"""Tests for morie.fn.nskov — Non-stationary kernel covariance."""

import numpy as np
import pytest

from morie.fn.nskov import nskov


class TestNskov:

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = nskov(coords, values)
        assert result["cov_matrix"].shape == (20, 20)
        assert result["n"] == 20

    def test_bisquare_kernel(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        values = rng.standard_normal(15)
        result = nskov(coords, values, kernel="bisquare")
        assert result["kernel"] == "bisquare"
        assert result["cov_matrix"].shape == (15, 15)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            nskov(np.ones((5, 2)), np.ones(3))

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            nskov(np.ones((5, 2)), np.ones(5), kernel="tophat")
