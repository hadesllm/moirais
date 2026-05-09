"""Tests for moirais.fn.stide — Spatio-temporal IDE model."""

import numpy as np
import pytest

from moirais.fn.stide import stide


class TestStide:

    def test_prediction_shape(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (8, 2))
        data = rng.standard_normal((5, 8))
        result = stide(coords, data)
        assert result["predictions"].shape == (4, 8)

    def test_kernel_row_sums(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (10, 2))
        data = rng.standard_normal((4, 10))
        result = stide(coords, data)
        row_sums = result["kernel_matrix"].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)

    def test_coords_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            stide(np.ones((5, 2)), np.ones((3, 4)))
