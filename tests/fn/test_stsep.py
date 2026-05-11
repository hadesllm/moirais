"""Tests for morie.fn.stsep — Separable spatio-temporal covariance."""

import numpy as np
import pytest

from morie.fn.stsep import stsep


class TestStsep:

    def test_output_shape(self):
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        times = np.array([0.0, 1.0, 2.0])
        result = stsep(coords, times)
        assert result["cov_matrix"].shape == (9, 9)
        assert result["spatial_cov"].shape == (3, 3)
        assert result["temporal_cov"].shape == (3, 3)

    def test_gaussian_model(self):
        coords = np.array([[0, 0], [1, 0]], dtype=float)
        times = np.array([0.0, 1.0])
        result = stsep(coords, times, model="gaussian")
        assert result["model"] == "gaussian"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            stsep(np.ones((3, 2)), np.ones(2), model="badmodel")

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 5, (4, 2))
        times = np.arange(3, dtype=float)
        result = stsep(coords, times)
        C = result["cov_matrix"]
        assert np.allclose(C, C.T, atol=1e-10)
