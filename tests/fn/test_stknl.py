"""Tests for moirais.fn.stknl — Spatio-temporal kernel smoothing."""

import numpy as np
import pytest

from moirais.fn.stknl import stknl


class TestStknl:

    def test_prediction_shape(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        times = np.arange(20, dtype=float)
        values = rng.standard_normal(20)
        tc = rng.uniform(2, 8, (5, 2))
        tt = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
        result = stknl(coords, times, values, tc, tt)
        assert result["predictions"].shape == (5,)

    def test_bandwidths_set(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        times = np.linspace(0, 10, 15)
        values = rng.standard_normal(15)
        result = stknl(coords, times, values, coords[:3], times[:3])
        assert result["spatial_bw"] > 0
        assert result["temporal_bw"] > 0

    def test_coords_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            stknl(np.ones((5, 2)), np.ones(5), np.ones(3), np.ones((1, 2)), np.ones(1))
