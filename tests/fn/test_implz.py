"""Tests for implz.py - Impulse response from I/O."""
import numpy as np
from moirais.fn.implz import impulse_from_io_fn, implz


def test_implz_returns_result():
    rng = np.random.default_rng(42)
    u = rng.standard_normal(256)
    h_true = np.array([1.0, 0.5, -0.3, 0.1])
    y = np.convolve(u, h_true)[:256]
    result = impulse_from_io_fn(u, y, N=32)
    assert result.name == "impulse_from_io"
    assert len(result.extra["impulse_response"]) == 32


def test_implz_alias():
    rng = np.random.default_rng(42)
    u = rng.standard_normal(64)
    y = rng.standard_normal(64)
    result = implz(u, y, N=16)
    assert result.name == "impulse_from_io"
