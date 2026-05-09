"""Tests for idwtf.py - Inverse DWT reconstruction."""
import numpy as np
from moirais.fn.dwtfn import dwt_decompose
from moirais.fn.idwtf import idwt_reconstruct, idwtf


def test_idwt_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(64)
    fwd = dwt_decompose(x, wavelet="haar", level=2)
    result = idwt_reconstruct(fwd.extra["coeffs"], wavelet="haar")
    assert result.name == "idwt_reconstruct"
    assert "signal" in result.extra


def test_idwt_roundtrip_haar():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    fwd = dwt_decompose(x, wavelet="haar", level=1)
    rec = idwt_reconstruct(fwd.extra["coeffs"], wavelet="haar")
    sig = rec.extra["signal"]
    assert len(sig) >= len(x)


def test_idwt_alias():
    x = np.random.default_rng(42).standard_normal(32)
    fwd = dwt_decompose(x, wavelet="haar", level=1)
    result = idwtf(fwd.extra["coeffs"], wavelet="haar")
    assert result.name == "idwt_reconstruct"
