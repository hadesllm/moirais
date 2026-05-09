"""Tests for cwtsc -- Continuous wavelet scalogram."""
import numpy as np
from moirais.fn.cwtsc import cwtsc
from moirais.fn._containers import DescriptiveResult


def test_cwtsc_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(256)
    result = cwtsc(x, fs=1000.0, n_scales=16)
    assert isinstance(result, DescriptiveResult)
    assert "scalogram" in result.extra


def test_cwtsc_shape():
    x = np.random.default_rng(7).standard_normal(128)
    result = cwtsc(x, fs=500.0, n_scales=10)
    S = result.extra["scalogram"]
    assert S.shape == (10, 128)


def test_cwtsc_nonnegative():
    x = np.sin(2 * np.pi * 50 * np.arange(200) / 1000)
    result = cwtsc(x, fs=1000.0, n_scales=8)
    assert np.all(result.extra["scalogram"] >= 0)
