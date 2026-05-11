"""Tests for morie.fn.medvt — median voter."""
import numpy as np
from morie.fn.medvt import medvt


def test_medvt_smoke():
    r = medvt([1, 2, 3, 4, 5])
    assert r.name == "median_voter"
    assert r.value == 3.0
    assert r.extra["n_voters"] == 5


def test_cheatsheet():
    from morie.fn.medvt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
