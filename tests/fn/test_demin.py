"""Tests for morie.fn.demin."""
import numpy as np
from morie.fn.demin import demin


def test_demin_smoke():
    rng = np.random.default_rng(42)
    result = demin(x=rng.uniform(40, 45, size=20), y=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.demin import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
