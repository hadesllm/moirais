"""Tests for moirais.fn.spgee — Spatial GEE."""

import numpy as np
import pytest

from moirais.fn.spgee import spgee


class TestSpgee:

    def test_gaussian_output(self):
        rng = np.random.default_rng(42)
        n = 30
        coords = rng.uniform(0, 10, (n, 2))
        X = np.column_stack([np.ones(n), coords[:, 0]])
        y = 2 + 3 * coords[:, 0] + rng.standard_normal(n) * 0.5
        result = spgee(coords, X, y, range_param=3.0)
        assert result["coefficients"].shape == (2,)
        assert result["robust_se"].shape == (2,)

    def test_sandwich_cov_shape(self):
        rng = np.random.default_rng(42)
        n = 20
        coords = rng.uniform(0, 10, (n, 2))
        X = np.column_stack([np.ones(n), coords[:, 0]])
        y = rng.standard_normal(n)
        result = spgee(coords, X, y)
        assert result["sandwich_cov"].shape == (2, 2)

    def test_row_mismatch_raises(self):
        with pytest.raises(ValueError, match="same number of rows"):
            spgee(np.ones((5, 2)), np.ones((3, 2)), np.ones(5))
