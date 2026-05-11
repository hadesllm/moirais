"""Tests for morie.fn.thetf."""
import numpy as np
from morie.fn.thetf import thetf


def test_thetf_smoke():
    rng = np.random.default_rng(42)
    result = thetf(y=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.thetf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
