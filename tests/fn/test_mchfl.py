"""Tests for mchfl -- Matched filter."""
import numpy as np
from moirais.fn.mchfl import matched_filter
from moirais.fn._containers import SignalResult


def test_matched_filter_basic():
    rng = np.random.default_rng(42)
    fs = 1000
    t = np.arange(0, 0.1, 1 / fs)
    template = np.sin(2 * np.pi * 50 * t)
    x = np.zeros(1000)
    x[300:300 + len(template)] = template
    x += rng.standard_normal(len(x)) * 0.1
    result = matched_filter(x, template, fs)
    assert isinstance(result, SignalResult)
    assert result.filtered is not None
    assert "peak_index" in result.extra
    assert "peak_snr" in result.extra


def test_matched_filter_finds_template():
    rng = np.random.default_rng(7)
    template = np.array([1, 2, 3, 2, 1], dtype=float)
    x = np.zeros(100)
    x[40:45] = template
    x += rng.standard_normal(len(x)) * 0.01
    result = matched_filter(x, template, fs=1.0)
    assert result.extra["peak_snr"] > 1.0


def test_matched_filter_snr_positive():
    template = np.array([0, 1, 0, -1, 0], dtype=float)
    x = np.zeros(50)
    x[20:25] = template * 5
    result = matched_filter(x, template)
    assert result.extra["peak_snr"] > 0
