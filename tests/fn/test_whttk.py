"""Tests for morie.fn.whttk."""
import numpy as np
from morie.fn.whttk import whttk


def test_whttk_smoke():
    rng = np.random.default_rng(42)
    result = whttk(y=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.whttk import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
