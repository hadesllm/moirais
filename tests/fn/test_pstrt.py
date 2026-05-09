"""Tests for moirais.fn.pstrt."""
import numpy as np
from moirais.fn.pstrt import pstrt


def test_pstrt_smoke():
    rng = np.random.default_rng(42)
    result = pstrt(ps=rng.uniform(0.1, 0.9, size=30))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.pstrt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
