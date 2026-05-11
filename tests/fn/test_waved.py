"""Tests for morie.fn.waved."""
import numpy as np
from morie.fn.waved import wave_1d


def test_waved_smoke():
    rng = np.random.default_rng(42)
    result = wave_1d(u0=np.sin(np.linspace(0, 4*np.pi, 100)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.waved import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
