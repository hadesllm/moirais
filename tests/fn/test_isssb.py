"""Tests for morie.fn.isssb — issue subset extract."""
import numpy as np
from morie.fn.isssb import isssb


def test_isssb_smoke():
    X = np.arange(20).reshape(4, 5).astype(float)
    r = isssb(X, [0, 2, 4])
    assert r.name == "issue_subset_extract"
    assert r.value == 3
    assert len(r.extra["subset"][0]) == 3


def test_cheatsheet():
    from morie.fn.isssb import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
