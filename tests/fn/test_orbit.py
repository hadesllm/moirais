"""Tests for morie.fn.orbit."""
import numpy as np
from morie.fn.orbit import kepler_orbit


def test_orbit_smoke():
    result = kepler_orbit(a=1.0, e=0.3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.orbit import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
