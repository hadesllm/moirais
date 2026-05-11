"""Tests for MSPE kriging."""
import numpy as np
from morie.fn.sgmspe import sgmspe


def test_sgmspe_smoke():
    kv = np.array([0.1, 0.2, 0.15, 0.3])
    r = sgmspe(kv)
    assert r.name == "mspe_kriging"
    assert abs(r.statistic - np.mean(kv)) < 1e-10
    assert "median" in r.extra
    assert "max" in r.extra


def test_cheatsheet():
    from morie.fn.sgmspe import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
