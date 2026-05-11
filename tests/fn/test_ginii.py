"""Tests for morie.fn.ginii."""
import numpy as np
from morie.fn.ginii import gini_coefficient


def test_ginii_smoke():
    rng = np.random.default_rng(42)
    result = gini_coefficient(incomes=rng.uniform(10, 100, size=50))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.ginii import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
