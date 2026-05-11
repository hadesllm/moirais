"""Tests for morsw.py - Generalized Morse wavelet."""
import numpy as np
from morie.fn.morsw import morse_wavelet, morsw


def test_morsw_returns_descriptive_result():
    result = morse_wavelet(beta=3, gamma_param=3, N=256)
    assert result.name == "morse_wavelet"
    assert "wavelet_freq" in result.extra
    assert "wavelet_time" in result.extra


def test_morsw_peak_frequency():
    result = morse_wavelet(beta=3, gamma_param=3)
    assert result.extra["peak_frequency"] > 0


def test_morsw_alias():
    result = morsw()
    assert result.name == "morse_wavelet"
