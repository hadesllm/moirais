"""Tests for wvthr.py - Wavelet threshold selection."""
import numpy as np
from morie.fn.wvthr import wavelet_threshold, wvthr


def test_wvthr_universal():
    c = np.random.default_rng(42).standard_normal(256)
    result = wavelet_threshold(c, method="universal")
    assert result.name == "wavelet_threshold"
    assert result.extra["threshold"] > 0


def test_wvthr_sure():
    c = np.random.default_rng(42).standard_normal(256)
    result = wavelet_threshold(c, method="sure")
    assert result.extra["threshold"] >= 0


def test_wvthr_alias():
    c = np.random.default_rng(42).standard_normal(64)
    result = wvthr(c)
    assert result.name == "wavelet_threshold"
