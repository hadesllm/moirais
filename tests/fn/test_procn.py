"""Tests for morie.fn.procn — Process convolution model."""

import numpy as np
import pytest

from morie.fn.procn import procn


class TestProcn:

    def test_fitted_shape(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (25, 2))
        values = rng.standard_normal(25)
        result = procn(coords, values, seed=42)
        assert result["fitted"].shape == (25,)
        assert result["residuals"].shape == (25,)

    def test_knots_count(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = procn(coords, values, n_knots=7, seed=42)
        assert len(result["knots"]) == 7

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            procn(np.ones((5, 2)), np.ones(3))
