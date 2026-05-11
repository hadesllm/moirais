"""Tests for instf.py - Instantaneous frequency."""
import numpy as np
from morie.fn.instf import instantaneous_freq, instf


def test_instf_returns_descriptive_result():
    t = np.linspace(0, 1, 256)
    x = np.sin(2 * np.pi * 10 * t)
    result = instantaneous_freq(x, fs=256.0)
    assert result.name == "instantaneous_freq"
    assert "inst_freq" in result.extra


def test_instf_sine_frequency():
    fs = 256.0
    f0 = 10.0
    t = np.arange(256) / fs
    x = np.sin(2 * np.pi * f0 * t)
    result = instantaneous_freq(x, fs=fs)
    mid = result.extra["inst_freq"][64:192]
    np.testing.assert_allclose(np.mean(mid), f0, atol=1.0)


def test_instf_alias():
    x = np.sin(np.linspace(0, 4 * np.pi, 64))
    result = instf(x)
    assert result.name == "instantaneous_freq"
