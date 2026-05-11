"""Tests for morie.fn.heckm."""
import numpy as np
from morie.fn.heckm import heckm


def test_heckm_smoke():
    rng = np.random.default_rng(42)
    n = 50
    z = rng.standard_normal((n, 2))
    selection = (z @ [1, 0.5] + rng.normal(0, 0.5, n)) > 0
    x = rng.standard_normal(n)
    y = 2 * x + rng.normal(0, 1, n)
    y[~selection] = np.nan
    result = heckm(y=y, x=x, z=z)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.heckm import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
