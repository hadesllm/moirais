"""Tests for morie.fn.crt."""
import numpy as np
from morie.fn.crt import crt


def test_crt_smoke():
    rng = np.random.default_rng(42)
    result = crt(remainders=[2, 3, 2], moduli=[3, 5, 7])
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.crt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
