"""Tests for morie.fn.cpvar."""
import numpy as np
from morie.fn.cpvar import cpvar


def test_cpvar_smoke():
    rng = np.random.default_rng(42)
    result = cpvar(y=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.cpvar import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
