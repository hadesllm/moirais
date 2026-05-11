"""Tests for morie.fn.jaccr."""
import numpy as np
from morie.fn.jaccr import jaccr


def test_jaccr_smoke():
    rng = np.random.default_rng(42)
    result = jaccr(set_a={1,2,3,4}, set_b={3,4,5,6})
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.jaccr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
