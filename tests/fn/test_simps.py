"""Tests for morie.fn.simps."""
import numpy as np
from morie.fn.simps import simps


def test_simps_smoke():
    rng = np.random.default_rng(42)
    result = simps(f=lambda x: x**2 - 2, a=0.0, b=1.0)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.simps import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
