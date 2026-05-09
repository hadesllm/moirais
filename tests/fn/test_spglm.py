"""Tests for moirais.fn.spglm — Spatial GLM."""

import numpy as np
import pytest

from moirais.fn.spglm import spglm


class TestSpglm:

    def test_gaussian_output(self):
        rng = np.random.default_rng(42)
        n = 30
        coords = rng.uniform(0, 10, (n, 2))
        X = np.column_stack([np.ones(n), coords[:, 0]])
        y = 2 + 3 * coords[:, 0] + rng.standard_normal(n) * 0.5
        result = spglm(coords, X, y, range_param=3.0)
        assert result["coefficients"].shape == (2,)
        assert result["fitted"].shape == (n,)

    def test_binomial_family(self):
        rng = np.random.default_rng(42)
        n = 25
        coords = rng.uniform(0, 10, (n, 2))
        X = np.column_stack([np.ones(n), coords[:, 0]])
        y = (rng.uniform(size=n) > 0.5).astype(float)
        result = spglm(coords, X, y, family="binomial", range_param=3.0)
        assert result["family"] == "binomial"

    def test_row_mismatch_raises(self):
        with pytest.raises(ValueError, match="X rows"):
            spglm(np.ones((5, 2)), np.ones((3, 2)), np.ones(5))
