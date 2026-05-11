"""Tests for morie.fn.csdnt."""
import numpy as np
from morie.fn.csdnt import csdnt


def test_csdnt_smoke():
    rng = np.random.default_rng(42)
    A = rng.standard_normal((20, 30))
    y = A @ rng.standard_normal(30) + rng.normal(0, 0.1, 20)
    result = csdnt(y=y, A=A)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.csdnt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
