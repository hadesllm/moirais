"""Tests for wveng.py - Wavelet energy distribution."""
import numpy as np
from morie.fn.wveng import wavelet_energy, wveng


def test_wveng_returns_descriptive_result():
    coeffs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    result = wavelet_energy(coeffs)
    assert result.name == "wavelet_energy"
    assert "energies" in result.extra


def test_wveng_relative_sums_to_one():
    coeffs = [np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5])]
    result = wavelet_energy(coeffs)
    rel = result.extra["relative_energies"]
    np.testing.assert_allclose(sum(rel), 1.0, atol=1e-10)


def test_wveng_alias():
    coeffs = [np.array([1.0]), np.array([2.0])]
    result = wveng(coeffs)
    assert result.name == "wavelet_energy"
