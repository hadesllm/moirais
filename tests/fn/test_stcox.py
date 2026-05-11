"""Tests for morie.fn.stcox."""
import numpy as np
from morie.fn.stcox import stcox


def test_stcox_smoke():
    rng = np.random.default_rng(42)
    result = stcox(
        x=rng.uniform(40, 45, size=20),
        y=rng.uniform(-80, -75, size=20),
        t=np.arange(20, dtype=float)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.stcox import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
