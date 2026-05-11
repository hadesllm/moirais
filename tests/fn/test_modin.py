"""Tests for morie.fn.modin."""
import numpy as np
from morie.fn.modin import modin


def test_modin_smoke():
    rng = np.random.default_rng(42)
    result = modin(a=12, m=7)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.modin import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
