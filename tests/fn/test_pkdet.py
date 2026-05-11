"""Tests for morie.fn.pkdet — advanced peak detection."""
import numpy as np
import pytest

from morie.fn.pkdet import peak_detect_advanced, pkdet


def test_sine_peaks():
    x = np.sin(np.linspace(0, 4 * np.pi, 500))
    result = peak_detect_advanced(x, prominence=0.3, distance=50)
    assert result.extra["n_peaks"] == 2


def test_no_peaks_flat():
    x = np.ones(100)
    result = peak_detect_advanced(x, prominence=0.5, distance=10)
    assert result.extra["n_peaks"] == 0


def test_high_prominence_filters():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    r_low = peak_detect_advanced(x, prominence=0.1, distance=5)
    r_high = peak_detect_advanced(x, prominence=2.0, distance=5)
    assert r_high.extra["n_peaks"] <= r_low.extra["n_peaks"]


def test_alias():
    assert pkdet is peak_detect_advanced
