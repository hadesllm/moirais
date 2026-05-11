"""Tests for morie.fn.lpflt."""
import numpy as np
from morie.fn.lpflt import lpflt


def test_lpflt_smoke():
    rng = np.random.default_rng(42)
    fs = 1000.0
    t = np.arange(0, 1.0, 1/fs)
    signal = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*200*t)
    result = lpflt(signal=signal, cutoff_hz=50.0, fs=fs)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.lpflt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
