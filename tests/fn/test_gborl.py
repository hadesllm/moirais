"""Tests for gborl.py - Gabor logon."""
import numpy as np
from moirais.fn.gborl import gabor_logon, gborl


def test_gborl_returns_descriptive_result():
    t = np.linspace(-1, 1, 256)
    result = gabor_logon(t, f0=10.0, sigma=0.1)
    assert result.name == "gabor_logon"
    assert "signal" in result.extra
    assert "envelope" in result.extra


def test_gborl_envelope_positive():
    t = np.linspace(-1, 1, 256)
    result = gabor_logon(t, f0=5.0, sigma=0.2)
    assert np.all(result.extra["envelope"] >= 0)


def test_gborl_alias():
    t = np.linspace(-1, 1, 64)
    result = gborl(t, f0=10.0, sigma=0.1)
    assert result.name == "gabor_logon"
