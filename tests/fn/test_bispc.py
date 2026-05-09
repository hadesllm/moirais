"""Tests for bispc.py - Bispectrum estimation."""
import numpy as np
from moirais.fn.bispc import bispectrum_fn, bispc


def test_bispc_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = bispectrum_fn(x, fs=100.0)
    assert result.name == "bispectrum"
    assert "bispectrum" in result.extra
    assert "frequencies" in result.extra


def test_bispc_matrix_square():
    x = np.random.default_rng(42).standard_normal(256)
    result = bispectrum_fn(x, nfft=128)
    bispec = result.extra["bispectrum"]
    assert bispec.shape == (64, 64)


def test_bispc_frequency_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = bispectrum_fn(x, nfft=256, fs=50.0)
    assert len(result.extra["frequencies"]) == 128


def test_bispc_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = bispc(x)
    assert result.name == "bispectrum"
