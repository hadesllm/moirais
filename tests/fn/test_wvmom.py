"""Tests for wvmom.py - Wavelet moments."""
import numpy as np
from morie.fn.wvmom import wavelet_moments, wvmom


def test_moments_returns_result():
    x = np.random.default_rng(42).standard_normal(128)
    result = wavelet_moments(x, level=3)
    assert result.name == "wavelet_moments"
    assert "moments" in result.extra


def test_moments_per_band():
    x = np.random.default_rng(42).standard_normal(256)
    result = wavelet_moments(x, level=3)
    moments = result.extra["moments"]
    assert len(moments) == 4
    for m in moments:
        assert "mean" in m
        assert "variance" in m
        assert "skewness" in m
        assert "kurtosis" in m


def test_moments_labels():
    x = np.random.default_rng(42).standard_normal(128)
    result = wavelet_moments(x, level=2)
    labels = result.extra["labels"]
    assert any(l.startswith("D") for l in labels)
    assert any(l.startswith("A") for l in labels)


def test_moments_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = wvmom(x, level=2)
    assert result.name == "wavelet_moments"
