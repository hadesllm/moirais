"""Tests for morie.fn.smoth -- epidemic smoothing."""

import numpy as np
import pytest
from morie.fn.smoth import epidemic_smooth


class TestEpiSmooth:
    def test_savgol_reduces_noise(self):
        rng = np.random.default_rng(42)
        clean = np.sin(np.linspace(0, 4 * np.pi, 100)) * 50 + 60
        noisy = clean + rng.normal(0, 10, 100)
        res = epidemic_smooth(noisy, method="savgol", window=7)
        resid_raw = np.std(noisy - clean)
        resid_sm = np.std(res["smoothed"] - clean)
        assert resid_sm < resid_raw

    def test_loess_nonnegative(self):
        rng = np.random.default_rng(42)
        inc = rng.poisson(5, 30).astype(float)
        res = epidemic_smooth(inc, method="loess", frac=0.3)
        assert np.all(res["smoothed"] >= 0)

    def test_residuals_sum(self):
        inc = np.array([10.0, 20.0, 30.0, 25.0, 15.0, 5.0])
        res = epidemic_smooth(inc, method="savgol", window=5)
        np.testing.assert_allclose(
            inc, res["smoothed"] + res["residuals"], atol=1e-10
        )

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            epidemic_smooth(np.ones(10), method="invalid")

    def test_short_array_raises(self):
        with pytest.raises(ValueError):
            epidemic_smooth(np.array([1.0, 2.0]))
