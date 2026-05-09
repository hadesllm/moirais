"""Tests for wvrec.py - Wavelet reconstruct."""
import numpy as np
from moirais.fn.wvdec import wavelet_decompose
from moirais.fn.wvrec import wavelet_reconstruct, wvrec


def test_reconstruct_returns_result():
    x = np.random.default_rng(42).standard_normal(128)
    dec = wavelet_decompose(x, wavelet="haar", level=2)
    result = wavelet_reconstruct(dec.extra["coeffs"], wavelet="haar")
    assert result.name == "wavelet_reconstruct"
    assert "reconstructed" in result.extra


def test_reconstruct_output_length():
    x = np.random.default_rng(42).standard_normal(64)
    dec = wavelet_decompose(x, wavelet="haar", level=2)
    rec = wavelet_reconstruct(dec.extra["coeffs"], wavelet="haar")
    recon = rec.extra["reconstructed"]
    assert len(recon) >= len(x)
    assert rec.extra["n_levels"] == 2


def test_reconstruct_alias():
    x = np.random.default_rng(42).standard_normal(64)
    dec = wavelet_decompose(x, wavelet="haar", level=1)
    result = wvrec(dec.extra["coeffs"], wavelet="haar")
    assert result.name == "wavelet_reconstruct"
