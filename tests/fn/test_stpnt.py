"""Tests for morie.fn.stpnt."""
import numpy as np
from morie.fn.stpnt import stpnt


def test_stpnt_smoke():
    rng = np.random.default_rng(42)
    result = stpnt(
        x=rng.uniform(40, 45, size=20),
        y=rng.uniform(-80, -75, size=20),
        t=np.arange(20, dtype=float)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.stpnt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
