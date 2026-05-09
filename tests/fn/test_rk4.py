"""Tests for moirais.fn.rk4."""
import numpy as np
from moirais.fn.rk4 import rk4


def test_rk4_smoke():
    result = rk4(f=lambda t, y: -0.5 * y, y0=np.array([1.0]), t_span=(0.0, 5.0))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.rk4 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
