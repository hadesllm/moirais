"""Tests for chird.py - Chirp detection."""
import numpy as np
from moirais.fn.chird import chirp_detect, chird


def test_chird_returns_descriptive_result():
    fs = 256.0
    t = np.arange(256) / fs
    x = np.cos(2 * np.pi * (10 * t + 20 * t**2))
    result = chirp_detect(x, fs=fs)
    assert result.name == "chirp_detect"
    assert "chirp_rate" in result.extra


def test_chird_detects_linear_sweep():
    fs = 512.0
    t = np.arange(512) / fs
    x = np.cos(2 * np.pi * (5 * t + 50 * t**2))
    result = chirp_detect(x, fs=fs)
    assert result.extra["chirp_rate"] > 0


def test_chird_alias():
    x = np.sin(np.linspace(0, 4 * np.pi, 128))
    result = chird(x)
    assert result.name == "chirp_detect"
