"""Tests for insft.py - Instantaneous frequency."""
import numpy as np
from moirais.fn.insft import instantaneous_freq, insft


def test_inst_freq_returns_result():
    t = np.linspace(0, 1, 500)
    x = np.sin(2 * np.pi * 10 * t)
    result = instantaneous_freq(x, fs=500.0)
    assert result.name == "instantaneous_freq"
    assert "inst_freq" in result.extra


def test_inst_freq_sine():
    fs = 1000.0
    t = np.linspace(0, 1, int(fs))
    freq = 50.0
    x = np.sin(2 * np.pi * freq * t)
    result = instantaneous_freq(x, fs=fs)
    mid = result.extra["inst_freq"][100:-100]
    assert abs(np.median(mid) - freq) < 5.0


def test_inst_freq_alias():
    x = np.random.default_rng(42).standard_normal(100)
    result = insft(x)
    assert result.name == "instantaneous_freq"
