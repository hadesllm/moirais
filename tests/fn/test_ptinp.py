"""Tests for morie.fn.ptinp."""
import numpy as np
from morie.fn.ptinp import ptinp


def test_ptinp_smoke():
    rng = np.random.default_rng(42)
    result = ptinp(
        point=(0.5, 0.5),
        polygon=np.array([[0,0],[2,0],[2,2],[0,2]], dtype=float)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.ptinp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
