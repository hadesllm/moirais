"""Tests for morie.fn.wavlt — wavelet decomposition."""
import numpy as np
from morie.fn.wavlt import wavelet_decompose


class TestWavelet:
    def test_basic(self):
        y = np.random.default_rng(42).standard_normal(64)
        res = wavelet_decompose(y)
        assert len(res.extra["details"]) > 0

    def test_reconstruction_approx(self):
        y = np.random.default_rng(42).standard_normal(16)
        res = wavelet_decompose(y, n_levels=1)
        approx = res.extra["approximation"]
        detail = res.extra["details"][0]
        recon = np.zeros(16)
        for i in range(8):
            recon[2*i] = (approx[i] + detail[i]) / np.sqrt(2)
            recon[2*i+1] = (approx[i] - detail[i]) / np.sqrt(2)
        np.testing.assert_allclose(recon, y, atol=1e-10)
