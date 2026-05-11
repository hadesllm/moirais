"""Tests for gbrwv -- Gabor-Wigner distribution."""
import numpy as np
from morie.fn.gbrwv import gbrwv
from morie.fn._containers import DescriptiveResult


def test_gbrwv_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(64)
    result = gbrwv(x, fs=100.0)
    assert isinstance(result, DescriptiveResult)
    assert "gw" in result.extra


def test_gbrwv_alpha_extremes():
    x = np.sin(2 * np.pi * 10 * np.arange(64) / 100)
    r0 = gbrwv(x, alpha=0.0)
    r1 = gbrwv(x, alpha=1.0)
    assert r0.extra["alpha"] == 0.0
    assert r1.extra["alpha"] == 1.0


def test_gbrwv_shape():
    x = np.random.default_rng(7).standard_normal(32)
    result = gbrwv(x, nfft=32)
    gw = result.extra["gw"]
    assert gw.shape == (32 // 2 + 1, 32)
