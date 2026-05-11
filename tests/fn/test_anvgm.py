"""Tests for morie.fn.anvgm — Anisotropic variogram."""

import numpy as np
import pytest

from morie.fn.anvgm import anvgm


class TestAnvgm:

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (30, 2))
        values = rng.standard_normal(30)
        result = anvgm(coords, values, n_directions=4, n_lags=8)
        assert result["semivariance"].shape == (4, 8)
        assert len(result["directions"]) == 4

    def test_directions_span_range(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (25, 2))
        values = rng.standard_normal(25)
        result = anvgm(coords, values, n_directions=6)
        assert result["directions"][0] == 0.0
        assert result["directions"][-1] < 180.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            anvgm(np.ones((5, 2)), np.ones(3))
