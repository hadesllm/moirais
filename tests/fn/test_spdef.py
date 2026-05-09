"""Tests for moirais.fn.spdef — Spatial deformation model."""

import numpy as np
import pytest

from moirais.fn.spdef import spdef


class TestSpdef:

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        values = rng.standard_normal(15)
        result = spdef(coords, values, seed=42)
        assert result["deformed_coords"].shape == (15, 2)
        assert result["stress"] >= 0.0

    def test_custom_dims(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = spdef(coords, values, n_dims=3, seed=42)
        assert result["deformed_coords"].shape == (20, 3)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            spdef(np.ones((5, 2)), np.ones(3))
