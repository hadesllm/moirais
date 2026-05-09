"""Tests for moirais.fn.ukrig — Universal kriging with trend."""

import numpy as np
import pytest

from moirais.fn.ukrig import ukrig


class TestUkrig:

    def test_prediction_shape(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = 2 * coords[:, 0] + rng.standard_normal(20) * 0.5
        target = rng.uniform(2, 8, (5, 2))
        result = ukrig(coords, values, target, range_param=5.0)
        assert result["predictions"].shape == (5,)
        assert result["variances"].shape == (5,)

    def test_variances_non_negative(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        values = rng.standard_normal(15)
        target = rng.uniform(2, 8, (3, 2))
        result = ukrig(coords, values, target)
        assert np.all(result["variances"] >= 0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            ukrig(np.ones((5, 2)), np.ones(3), np.ones((1, 2)))
