"""Tests for moirais.fn.ginii."""
import numpy as np
from moirais.fn.ginii import gini_coefficient


def test_ginii_smoke():
    rng = np.random.default_rng(42)
    result = gini_coefficient(incomes=rng.uniform(10, 100, size=50))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.ginii import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
