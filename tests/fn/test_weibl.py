"""Tests for moirais.fn.weibl."""
import numpy as np
from moirais.fn.weibl import weibl


def test_weibl_smoke():
    rng = np.random.default_rng(42)
    data = rng.exponential(scale=10.0, size=50)
    result = weibl(data=data)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.weibl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
