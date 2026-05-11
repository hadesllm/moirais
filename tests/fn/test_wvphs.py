"""Tests for wvphs.py - Wavelet phase."""
import numpy as np
from morie.fn.wvphs import wavelet_phase, wvphs


def test_wvphs_returns_descriptive_result():
    coeffs = np.exp(1j * np.linspace(0, 4 * np.pi, 64))
    result = wavelet_phase(coeffs)
    assert result.name == "wavelet_phase"
    assert "phase" in result.extra


def test_wvphs_unwrapped_monotonic():
    coeffs = np.exp(1j * np.linspace(0, 6 * np.pi, 128))
    result = wavelet_phase(coeffs)
    phase = result.extra["phase"]
    assert np.all(np.diff(phase) > 0)


def test_wvphs_alias():
    coeffs = np.exp(1j * np.linspace(0, 2 * np.pi, 32))
    result = wvphs(coeffs)
    assert result.name == "wavelet_phase"
