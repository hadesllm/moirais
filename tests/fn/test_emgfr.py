"""Tests for emgfr -- EMG median/mean frequency."""
import numpy as np
from morie.fn.emgfr import emgfr
from morie.fn._containers import DescriptiveResult


def test_emgfr_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1024)
    result = emgfr(x, fs=1000.0)
    assert isinstance(result, DescriptiveResult)
    assert "median_freq" in result.extra
    assert "mean_freq" in result.extra


def test_emgfr_tone():
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 200 * t)
    result = emgfr(x, fs=fs)
    assert abs(result.extra["median_freq"] - 200) < 50


def test_emgfr_positive_power():
    x = np.random.default_rng(7).standard_normal(512)
    result = emgfr(x)
    assert result.extra["total_power"] > 0
