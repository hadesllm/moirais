"""Tests for moirais.fn.nscov — Non-stationary covariance function."""

import numpy as np
import pytest

from moirais.fn.nscov import nscov


class TestNscov:

    def test_output_shape(self):
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        result = nscov(coords)
        assert result["cov_matrix"].shape == (4, 4)
        assert result["n"] == 4

    def test_constant_params(self):
        coords = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        result = nscov(coords, sigma_func="constant", range_func="constant")
        assert np.allclose(result["local_sill"], 1.0)
        assert np.allclose(result["local_range"], 1.0)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 5, (10, 2))
        result = nscov(coords)
        cov = result["cov_matrix"]
        assert np.allclose(cov, cov.T, atol=1e-10)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            nscov(np.ones((5, 3)))
