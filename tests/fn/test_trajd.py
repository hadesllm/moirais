"""Tests for morie.fn.trajd."""
import numpy as np
from morie.fn.trajd import trajectory_distance


def test_trajd_smoke():
    rng = np.random.default_rng(42)
    result = trajectory_distance(path_a=rng.uniform(size=(10, 2)), path_b=rng.uniform(size=(10, 2)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.trajd import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
