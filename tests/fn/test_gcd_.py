"""Tests for morie.fn.gcd_."""
import numpy as np
from morie.fn.gcd_ import gcd_


def test_gcd__smoke():
    rng = np.random.default_rng(42)
    result = gcd_(a=12, b=8)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.gcd_ import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
