"""Tests for morie.fn.rspln."""
import numpy as np
from morie.fn.rspln import restricted_cubic_spline


def test_rspln_smoke():
    rng = np.random.default_rng(42)
    result = restricted_cubic_spline(x=rng.standard_normal(20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.rspln import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
