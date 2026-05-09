"""Tests for spwvd.py - Smoothed pseudo Wigner-Ville distribution."""
import numpy as np
from moirais.fn.spwvd import smoothed_pseudo_wvd_fn, spwvd


def test_spwvd_returns_descriptive_result():
    x = np.sin(2 * np.pi * 5 * np.arange(32) / 32)
    result = smoothed_pseudo_wvd_fn(x, fs=32.0)
    assert result.name == "smoothed_pseudo_wvd"
    assert "tfd" in result.extra


def test_spwvd_tfd_shape():
    n = 16
    x = np.random.default_rng(42).standard_normal(n)
    result = smoothed_pseudo_wvd_fn(x)
    tfd = result.extra["tfd"]
    assert tfd.shape == (n, n)


def test_spwvd_time_length():
    n = 24
    x = np.random.default_rng(42).standard_normal(n)
    result = smoothed_pseudo_wvd_fn(x, fs=100.0)
    assert len(result.extra["time"]) == n


def test_spwvd_alias():
    x = np.random.default_rng(42).standard_normal(16)
    result = spwvd(x)
    assert result.name == "smoothed_pseudo_wvd"
