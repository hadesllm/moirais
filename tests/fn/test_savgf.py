"""Tests for morie.fn.savgf."""
import numpy as np
from morie.fn.savgf import savgf


def test_savgf_smoke():
    rng = np.random.default_rng(42)
    result = savgf(y=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.savgf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
