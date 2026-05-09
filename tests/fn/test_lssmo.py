"""Tests for moirais.fn.lssmo."""
import numpy as np
from moirais.fn.lssmo import lssmo


def test_lssmo_smoke():
    rng = np.random.default_rng(42)
    result = lssmo(x=rng.uniform(40, 45, size=20), y=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.lssmo import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
