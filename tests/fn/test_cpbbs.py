"""Tests for moirais.fn.cpbbs."""
import numpy as np
from moirais.fn.cpbbs import cpbbs


def test_cpbbs_smoke():
    rng = np.random.default_rng(42)
    result = cpbbs(y=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.cpbbs import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
