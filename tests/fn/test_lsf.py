"""Tests for lsf.py - Line Spectral Frequencies."""
import numpy as np
from morie.fn.lsf import line_spectral_freq_fn, lsf


def test_lsf_returns_result():
    coeffs = np.array([0.5, -0.3, 0.1, -0.05])
    result = line_spectral_freq_fn(coeffs)
    assert result.name == "line_spectral_freq"
    assert "lsf" in result.extra


def test_lsf_values_in_range():
    coeffs = np.array([0.3, -0.2])
    result = line_spectral_freq_fn(coeffs)
    assert np.all(result.extra["lsf"] >= 0)
    assert np.all(result.extra["lsf"] <= np.pi)


def test_lsf_alias():
    coeffs = np.array([0.5, -0.3])
    result = lsf(coeffs)
    assert result.name == "line_spectral_freq"
