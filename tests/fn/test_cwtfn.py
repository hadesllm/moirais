"""Tests for cwtfn.py - Continuous wavelet transform."""
import numpy as np
from moirais.fn.cwtfn import cwt_compute_fn, cwtfn


def test_cwt_returns_descriptive_result():
    x = np.sin(2 * np.pi * 10 * np.arange(256) / 256)
    result = cwt_compute_fn(x, fs=256.0)
    assert result.name == "cwt"
    assert "coefficients" in result.extra
    assert "scales" in result.extra


def test_cwt_coefficient_shape():
    x = np.random.default_rng(42).standard_normal(128)
    scales = np.arange(1, 17)
    result = cwt_compute_fn(x, scales=scales)
    coeffs = result.extra["coefficients"]
    assert coeffs.shape == (16, 128)


def test_cwt_frequencies_positive():
    x = np.random.default_rng(42).standard_normal(128)
    result = cwt_compute_fn(x, fs=100.0)
    freqs = result.extra["frequencies"]
    assert all(f > 0 for f in freqs)


def test_cwt_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = cwtfn(x, wavelet="mexican_hat")
    assert result.name == "cwt"
