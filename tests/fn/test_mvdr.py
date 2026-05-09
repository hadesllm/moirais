"""Tests for mvdr.py - MVDR (Capon) spectrum."""
import numpy as np
from moirais.fn.mvdr import mvdr_spectrum_fn, mvdr


def test_mvdr_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = mvdr_spectrum_fn(x, order=8, nfft=64)
    assert result.name == "mvdr_spectrum"
    assert len(result.extra["frequencies"]) == 64
    assert len(result.extra["psd"]) == 64


def test_mvdr_psd_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = mvdr_spectrum_fn(x, order=8, nfft=64)
    assert np.all(result.extra["psd"] > 0)


def test_mvdr_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = mvdr(x, order=4, nfft=32)
    assert result.name == "mvdr_spectrum"
