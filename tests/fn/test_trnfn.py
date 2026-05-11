"""Tests for morie.fn.trnfn."""
import numpy as np
from morie.fn.trnfn import trnfn


def test_trnfn_smoke():
    rng = np.random.default_rng(42)
    result = trnfn(num=rng.standard_normal(20), den=rng.standard_normal(20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.trnfn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
