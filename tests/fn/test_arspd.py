"""Tests for arspd.py - AR spectrum."""
import numpy as np
from morie.fn.arspd import ar_spectrum_fn, arspd


def test_arspd_returns_result():
    ar = np.array([0.5, -0.3])
    result = ar_spectrum_fn(ar, sigma2=1.0, nfft=128)
    assert result.name == "ar_spectrum"
    assert "frequencies" in result.extra
    assert "psd" in result.extra
    assert len(result.extra["frequencies"]) == 128


def test_arspd_psd_positive():
    ar = np.array([0.5, -0.3])
    result = ar_spectrum_fn(ar, sigma2=1.0)
    assert np.all(result.extra["psd"] > 0)


def test_arspd_alias():
    ar = np.array([0.3])
    result = arspd(ar, sigma2=0.5, nfft=64)
    assert result.name == "ar_spectrum"
