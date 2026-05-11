"""Tests for morie.fn.linit."""
import numpy as np
from morie.fn.linit import linit


def test_linit_smoke():
    rng = np.random.default_rng(42)
    result = linit(p1=(0.0, 1.0), p2=(0.0, 1.0), p3=(0.0, 1.0), p4=(0.0, 1.0))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.linit import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
