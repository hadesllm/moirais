"""Tests for bspec -- Bispectrum estimation."""
import numpy as np
from morie.fn.bspec import bspec
from morie.fn._containers import DescriptiveResult


def test_bspec_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(512)
    result = bspec(x, fs=1000.0, nfft=64)
    assert isinstance(result, DescriptiveResult)
    assert "bispectrum" in result.extra


def test_bspec_shape():
    x = np.random.default_rng(7).standard_normal(256)
    result = bspec(x, nfft=32)
    bisp = result.extra["bispectrum"]
    nf = 32 // 2 + 1
    assert bisp.shape == (nf, nf)


def test_bspec_nonzero_for_nonlinear():
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 100 * t) ** 2
    result = bspec(x, fs=fs, nfft=128)
    assert result.value > 0
