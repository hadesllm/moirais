"""Tests for morie.fn.semea."""
import numpy as np
from morie.fn.semea import semea


def test_semea_smoke():
    rng = np.random.default_rng(42)
    result = semea(sd=1.5, reliability=0.85)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.semea import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
