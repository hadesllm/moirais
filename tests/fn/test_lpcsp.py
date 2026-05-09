"""Tests for lpcsp.py - LPC spectrum."""
import numpy as np
from moirais.fn.lpcsp import lpc_spectrum_fn, lpcsp


def test_lpcsp_returns_result():
    coeffs = np.array([0.5, -0.3, 0.1])
    result = lpc_spectrum_fn(coeffs, sigma2=1.0, nfft=128)
    assert result.name == "lpc_spectrum"
    assert len(result.extra["frequencies"]) == 128
    assert len(result.extra["psd"]) == 128


def test_lpcsp_psd_positive():
    coeffs = np.array([0.5, -0.3])
    result = lpc_spectrum_fn(coeffs, sigma2=0.5)
    assert np.all(result.extra["psd"] > 0)


def test_lpcsp_alias():
    coeffs = np.array([0.3])
    result = lpcsp(coeffs, sigma2=1.0, nfft=64)
    assert result.name == "lpc_spectrum"
