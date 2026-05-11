"""Tests for arspc.py - AR model spectrum."""
import numpy as np
import pytest
from morie.fn.arspc import ar_spectrum_fn, arspc


def test_ar_spectrum_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_spectrum_fn(x)
    assert result.name == "ar_spectrum"
    assert "freqs" in result.extra
    assert "psd" in result.extra


def test_ar_spectrum_psd_nonnegative():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_spectrum_fn(x, fs=100.0)
    assert np.all(result.extra["psd"] >= 0)


def test_ar_spectrum_freq_range():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_spectrum_fn(x, fs=100.0, n_points=512)
    freqs = result.extra["freqs"]
    assert freqs[0] >= 0
    assert freqs[-1] <= 50.0


def test_arspc_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = arspc(x)
    assert result.name == "ar_spectrum"
