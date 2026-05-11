"""Tests for insta.py - Instantaneous amplitude."""
import numpy as np
from morie.fn.insta import instantaneous_amp, insta


def test_insta_returns_descriptive_result():
    x = np.sin(np.linspace(0, 4 * np.pi, 128))
    result = instantaneous_amp(x)
    assert result.name == "instantaneous_amp"
    assert "envelope" in result.extra


def test_insta_envelope_positive():
    x = np.sin(np.linspace(0, 4 * np.pi, 128))
    result = instantaneous_amp(x)
    assert np.all(result.extra["envelope"] >= 0)


def test_insta_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = insta(x)
    assert result.name == "instantaneous_amp"
