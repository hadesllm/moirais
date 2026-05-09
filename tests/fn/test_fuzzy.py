"""Tests for moirais.fn.fuzzy."""
import numpy as np
from moirais.fn.fuzzy import fuzzy


def test_fuzzy_smoke():
    rng = np.random.default_rng(42)
    result = fuzzy(x=rng.standard_normal(20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.fuzzy import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
