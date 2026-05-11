"""Tests for esprt.py - ESPRIT frequency estimation."""
import numpy as np
from morie.fn.esprt import esprit_freq_fn, esprt


def test_esprt_returns_result():
    rng = np.random.default_rng(42)
    t = np.arange(256) / 100.0
    x = np.sin(2 * np.pi * 10 * t) + 0.1 * rng.standard_normal(256)
    result = esprit_freq_fn(x, nsources=1, order=8, fs=100.0)
    assert result.name == "esprit_freq"
    assert "frequencies" in result.extra


def test_esprt_frequency_count():
    rng = np.random.default_rng(42)
    t = np.arange(256) / 100.0
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 25 * t) + 0.1 * rng.standard_normal(256)
    result = esprit_freq_fn(x, nsources=2, order=8, fs=100.0)
    assert len(result.extra["frequencies"]) == 2


def test_esprt_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = esprt(x, nsources=1, order=4)
    assert result.name == "esprit_freq"
