"""Tests for morie.fn.polar."""
import numpy as np
from morie.fn.polar import polar_decompose


def test_polar_smoke():
    rng = np.random.default_rng(42)
    result = polar_decompose(A=rng.standard_normal((4, 4)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.polar import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
