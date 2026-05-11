"""Tests for morie.fn.xvgm — Cross-variogram estimation."""

import numpy as np
import pytest

from morie.fn.xvgm import xvgm


class TestXvgm:

    def test_output_keys(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        v1 = rng.standard_normal(20)
        v2 = v1 * 0.8 + rng.standard_normal(20) * 0.2
        result = xvgm(coords, v1, v2, n_lags=8)
        assert len(result["lags"]) == 8
        assert len(result["counts"]) == 8

    def test_cross_sv_populated(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (25, 2))
        v1 = rng.standard_normal(25)
        v2 = rng.standard_normal(25)
        result = xvgm(coords, v1, v2)
        assert np.any(~np.isnan(result["cross_semivariance"]))

    def test_var2_length_mismatch(self):
        with pytest.raises(ValueError, match="var2 must have same length"):
            xvgm(np.ones((5, 2)), np.ones(5), np.ones(3))
