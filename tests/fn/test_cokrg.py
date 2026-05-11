"""Tests for morie.fn.cokrg — Co-kriging."""

import numpy as np
import pytest

from morie.fn.cokrg import cokrg


class TestCokrg:

    def test_prediction_shape(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        primary = rng.standard_normal(20)
        secondary = primary * 0.8 + rng.standard_normal(20) * 0.2
        target = rng.uniform(2, 8, (5, 2))
        result = cokrg(coords, primary, secondary, target)
        assert result["predictions"].shape == (5,)

    def test_variances_non_negative(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        p = rng.standard_normal(15)
        s = rng.standard_normal(15)
        target = rng.uniform(2, 8, (3, 2))
        result = cokrg(coords, p, s, target)
        assert np.all(result["variances"] >= 0)

    def test_secondary_length_mismatch(self):
        with pytest.raises(ValueError, match="secondary must have same length"):
            cokrg(np.ones((5, 2)), np.ones(5), np.ones(3), np.ones((1, 2)))
