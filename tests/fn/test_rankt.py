"""Tests for morie.fn.rankt."""
import numpy as np
from morie.fn.rankt import rank_transform


def test_rankt_smoke():
    rng = np.random.default_rng(42)
    result = rank_transform(x=rng.standard_normal(20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.rankt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
