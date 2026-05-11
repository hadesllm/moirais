"""Tests for morie.fn.lufac."""
import numpy as np
from morie.fn.lufac import lu_factorize


def test_lufac_smoke():
    rng = np.random.default_rng(42)
    result = lu_factorize(A=rng.standard_normal((4, 4)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.lufac import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
