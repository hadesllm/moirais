"""Tests for moirais.fn.knspl — Kernel non-stationary spatial prediction."""

import numpy as np
import pytest

from moirais.fn.knspl import knspl


class TestKnspl:

    def test_prediction_shape(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        target = rng.uniform(2, 8, (5, 2))
        result = knspl(coords, values, target)
        assert result["predictions"].shape == (5,)

    def test_interpolation_at_data(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        values = rng.standard_normal(15)
        result = knspl(coords, values, coords, bandwidth=0.01)
        assert np.allclose(result["predictions"], values, atol=0.5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            knspl(np.ones((5, 2)), np.ones(3), np.ones((2, 2)))

    def test_unknown_kernel_raises(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (10, 2))
        with pytest.raises(ValueError, match="Unknown kernel"):
            knspl(coords, np.ones(10), np.ones((1, 2)), kernel="bad")
