"""Tests for morie.fn.plakb."""
import numpy as np
from morie.fn.plakb import plakb


def test_plakb_smoke():
    rng = np.random.default_rng(42)
    result = plakb(n_factors=3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.plakb import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
