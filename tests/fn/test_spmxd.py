"""Tests for moirais.fn.spmxd — Spatial mixed model."""

import numpy as np
import pytest

from moirais.fn.spmxd import spmxd


class TestSpmxd:

    def test_output_keys(self):
        rng = np.random.default_rng(42)
        n = 25
        coords = rng.uniform(0, 10, (n, 2))
        X = np.column_stack([np.ones(n), coords[:, 0]])
        y = 2 + 3 * coords[:, 0] + rng.standard_normal(n) * 0.5
        result = spmxd(coords, X, y, range_param=3.0)
        assert result["beta"].shape == (2,)
        assert result["random_effects"].shape == (n,)
        assert result["log_likelihood"] != 0.0

    def test_fitted_residuals(self):
        rng = np.random.default_rng(42)
        n = 20
        coords = rng.uniform(0, 10, (n, 2))
        X = np.column_stack([np.ones(n), coords[:, 0]])
        y = rng.standard_normal(n)
        result = spmxd(coords, X, y)
        recon = result["fitted"] + result["residuals"]
        assert np.allclose(recon, y, atol=1e-6)

    def test_row_mismatch_raises(self):
        with pytest.raises(ValueError, match="same number of rows"):
            spmxd(np.ones((5, 2)), np.ones((3, 2)), np.ones(5))
