"""Tests for morie.fn.mankt."""
import numpy as np
from morie.fn.mankt import mankt


def test_mankt_smoke():
    rng = np.random.default_rng(42)
    result = mankt(
        dist_x=np.abs(rng.standard_normal((10, 10))) + np.eye(10),
        dist_y=np.abs(rng.standard_normal((10, 10))) + np.eye(10)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.mankt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
