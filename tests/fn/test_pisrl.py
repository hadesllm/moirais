"""Tests for pisrl.py - Pisarenko harmonic decomposition."""
import numpy as np
from morie.fn.pisrl import pisarenko_fn, pisrl


def test_pisrl_returns_result():
    rng = np.random.default_rng(42)
    t = np.arange(256) / 100.0
    x = np.sin(2 * np.pi * 10 * t) + 0.1 * rng.standard_normal(256)
    result = pisarenko_fn(x, nsources=1, fs=100.0)
    assert result.name == "pisarenko"
    assert "frequencies" in result.extra


def test_pisrl_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = pisrl(x, nsources=1)
    assert result.name == "pisarenko"
