"""Tests for moirais.fn.cpelt."""
import numpy as np
from moirais.fn.cpelt import cpelt


def test_cpelt_smoke():
    rng = np.random.default_rng(42)
    result = cpelt(y=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.cpelt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
