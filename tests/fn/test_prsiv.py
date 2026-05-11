"""Tests for morie.fn.prsiv."""
import numpy as np
from morie.fn.prsiv import prsiv


def test_prsiv_smoke():
    rng = np.random.default_rng(42)
    result = prsiv(n=5)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.prsiv import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
