"""Tests for morie.fn.selbf."""
import numpy as np
from morie.fn.selbf import selbf


def test_selbf_smoke():
    rng = np.random.default_rng(42)
    result = selbf(p_selected=0.5, outcome_diff=0.3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.selbf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
