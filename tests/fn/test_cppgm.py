"""Tests for moirais.fn.cppgm."""
import numpy as np
from moirais.fn.cppgm import cppgm


def test_cppgm_smoke():
    rng = np.random.default_rng(42)
    result = cppgm(counts=rng.integers(0, 10, size=20).astype(float))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.cppgm import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
