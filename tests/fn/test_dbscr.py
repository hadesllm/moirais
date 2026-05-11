"""Tests for morie.fn.dbscr."""
import numpy as np
from morie.fn.dbscr import dbscr


def test_dbscr_smoke():
    rng = np.random.default_rng(42)
    result = dbscr(X=rng.standard_normal((30, 3)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.dbscr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
