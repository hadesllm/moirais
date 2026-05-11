"""Tests for morie.fn.delny."""
import numpy as np
from morie.fn.delny import delny


def test_delny_smoke():
    rng = np.random.default_rng(42)
    result = delny(points=rng.uniform(size=(20, 2)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.delny import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
