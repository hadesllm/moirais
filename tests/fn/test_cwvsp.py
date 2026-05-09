"""Tests for cwvsp.py - CWT spectrum."""
import numpy as np
from moirais.fn.cwvsp import cwt_spectrum, cwvsp


def test_cwt_returns_result():
    t = np.linspace(0, 1, 256)
    x = np.sin(2 * np.pi * 10 * t)
    result = cwt_spectrum(x, scales=np.arange(1, 32))
    assert result.name == "cwt_spectrum"
    assert "power" in result.extra


def test_cwt_auto_scales():
    x = np.random.default_rng(42).standard_normal(128)
    result = cwt_spectrum(x)
    assert result.extra["power"].shape[1] == 128


def test_cwt_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = cwvsp(x, scales=np.arange(1, 16))
    assert result.name == "cwt_spectrum"
