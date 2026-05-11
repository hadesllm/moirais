"""Tests for morie.fn.sintf."""
import numpy as np
from morie.fn.sintf import sintf


def test_sintf_smoke():
    rng = np.random.default_rng(42)
    points = rng.uniform(size=(20, 2))
    values = np.sin(points[:, 0]) + np.cos(points[:, 1])
    query = rng.uniform(0.2, 0.8, size=(5, 2))
    result = sintf(points=points, values=values, query=query)
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.sintf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
