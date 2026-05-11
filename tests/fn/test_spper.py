"""Tests for morie.fn.spper — Spatial periodogram."""

import numpy as np
import pytest

from morie.fn.spper import spper


class TestSpper:

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = spper(coords, values, n_freq=15)
        assert result["power"].shape == (15, 15)
        assert len(result["freq_x"]) == 15

    def test_power_non_negative(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        values = rng.standard_normal(15)
        result = spper(coords, values)
        assert np.all(result["power"] >= 0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            spper(np.ones((5, 2)), np.ones(3))
