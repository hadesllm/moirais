"""Tests for wvcoh.py - Wavelet coherence."""
import numpy as np
from morie.fn.wvcoh import wavelet_coherence, wvcoh


def test_wvcoh_returns_descriptive_result():
    rng = np.random.default_rng(42)
    x = np.sin(np.linspace(0, 4 * np.pi, 128))
    y = np.sin(np.linspace(0, 4 * np.pi, 128)) + 0.1 * rng.standard_normal(128)
    result = wavelet_coherence(x, y, fs=128.0)
    assert result.name == "wavelet_coherence"
    assert "coherence" in result.extra


def test_wvcoh_identical_signals():
    x = np.sin(np.linspace(0, 4 * np.pi, 64))
    result = wavelet_coherence(x, x, fs=64.0)
    assert result.value > 0.5


def test_wvcoh_alias():
    x = np.random.default_rng(42).standard_normal(64)
    y = np.random.default_rng(43).standard_normal(64)
    result = wvcoh(x, y)
    assert result.name == "wavelet_coherence"
