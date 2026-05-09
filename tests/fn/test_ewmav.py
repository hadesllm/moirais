"""Tests for moirais.fn.ewmav."""
import numpy as np
from moirais.fn.ewmav import ewmav


def test_ewmav_smoke():
    rng = np.random.default_rng(42)
    result = ewmav(returns=rng.uniform(10, 100, size=50))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.ewmav import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
