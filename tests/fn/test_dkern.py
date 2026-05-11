"""Tests for morie.fn.dkern — Deformation kernel."""

import numpy as np
import pytest

from morie.fn.dkern import dkern


class TestDkern:

    def test_output_keys(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = dkern(coords, values, seed=42)
        assert "deformed_coords" in result
        assert "knots" in result
        assert result["deformed_coords"].shape == (20, 2)

    def test_n_basis_respected(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (30, 2))
        values = rng.standard_normal(30)
        result = dkern(coords, values, n_basis=8, seed=42)
        assert result["n_basis"] == 8

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            dkern(np.ones((5, 3)), np.ones(5))
