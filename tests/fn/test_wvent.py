"""Tests for wvent.py - Wavelet entropy."""
import numpy as np
from moirais.fn.wvent import wavelet_entropy, wvent


def test_wvent_returns_descriptive_result():
    coeffs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    result = wavelet_entropy(coeffs)
    assert result.name == "wavelet_entropy"
    assert result.extra["entropy"] >= 0


def test_wvent_uniform_max_entropy():
    coeffs = [np.ones(10), np.ones(10)]
    result = wavelet_entropy(coeffs)
    assert result.extra["entropy"] > 0


def test_wvent_alias():
    coeffs = [np.array([1.0]), np.array([2.0])]
    result = wvent(coeffs)
    assert result.name == "wavelet_entropy"
