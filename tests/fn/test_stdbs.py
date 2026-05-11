"""Tests for morie.fn.stdbs."""
import numpy as np
from morie.fn.stdbs import st_dbscan


def test_stdbs_smoke():
    rng = np.random.default_rng(42)
    result = st_dbscan(
        coords=rng.uniform(size=(20, 2)),
        times=np.arange(20, dtype=float),
        eps_spatial=1.0,
        eps_temporal=1.0
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.stdbs import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
