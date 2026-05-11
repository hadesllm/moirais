"""Tests for morie.fn.stfft."""
import numpy as np
from morie.fn.stfft import stft


def test_stfft_smoke():
    rng = np.random.default_rng(42)
    result = stft(signal=np.sin(np.linspace(0, 4*np.pi, 100)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.stfft import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
