"""Tests for moirais.fn.lyapn."""
import numpy as np
from moirais.fn.lyapn import lyapn


def test_lyapn_smoke():
    A = np.array([[-1, 0], [0, -2]], dtype=float)
    result = lyapn(A=A)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.lyapn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
