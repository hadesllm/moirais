"""Tests for moirais.fn.holts."""
import numpy as np
from moirais.fn.holts import holts


def test_holts_smoke():
    rng = np.random.default_rng(42)
    result = holts(y=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.holts import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
