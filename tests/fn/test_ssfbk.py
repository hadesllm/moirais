"""Tests for moirais.fn.ssfbk."""
import numpy as np
from moirais.fn.ssfbk import ssfbk


def test_ssfbk_smoke():
    A = np.array([[0, 1], [-2, -3]], dtype=float)
    B = np.array([[0], [1]], dtype=float)
    poles = np.array([-1, -2])
    result = ssfbk(A=A, B=B, poles=poles)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.ssfbk import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
