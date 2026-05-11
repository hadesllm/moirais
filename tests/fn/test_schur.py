"""Tests for morie.fn.schur."""
import numpy as np
from morie.fn.schur import schur_decompose


def test_schur_smoke():
    rng = np.random.default_rng(42)
    result = schur_decompose(A=rng.standard_normal((4, 4)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.schur import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
