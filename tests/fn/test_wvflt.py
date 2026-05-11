"""Tests for wvflt.py - Wavelet filtering."""
import numpy as np
from morie.fn.wvflt import wavelet_filter, wvflt


def test_filter_approx():
    x = np.random.default_rng(42).standard_normal(128)
    result = wavelet_filter(x, level=2, keep="approx")
    assert result.name == "wavelet_filter"
    assert result.extra["keep"] == "approx"
    assert len(result.extra["filtered"]) == len(x)


def test_filter_detail():
    x = np.random.default_rng(42).standard_normal(128)
    result = wavelet_filter(x, level=2, keep="detail")
    assert result.extra["keep"] == "detail"


def test_filter_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = wvflt(x, level=2)
    assert result.name == "wavelet_filter"
