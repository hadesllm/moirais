"""Tests for morie.fn.hrsde."""
import numpy as np
from morie.fn.hrsde import home_range_kde


def test_hrsde_smoke():
    rng = np.random.default_rng(42)
    result = home_range_kde(coords=rng.uniform(size=(20, 2)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.hrsde import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
