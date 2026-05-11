"""Tests for morie.fn.cvar."""
import numpy as np
from morie.fn.cvar import cvar


def test_cvar_smoke():
    rng = np.random.default_rng(42)
    result = cvar(returns=rng.uniform(10, 100, size=50))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.cvar import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
