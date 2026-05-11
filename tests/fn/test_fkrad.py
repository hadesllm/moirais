"""Tests for morie.fn.fkrad."""
import numpy as np
from morie.fn.fkrad import fkrad


def test_fkrad_smoke():
    rng = np.random.default_rng(42)
    result = fkrad(text="The quick brown fox jumps over the lazy dog")
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.fkrad import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
