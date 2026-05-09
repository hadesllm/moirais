"""Tests for moirais.fn.spric."""
import numpy as np
from moirais.fn.spric import spric


def test_spric_smoke():
    samples = np.array([20, 15, 10, 5, 3, 2, 1, 1])
    result = spric(samples=samples)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.spric import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
