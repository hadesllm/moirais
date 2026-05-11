"""Tests for normal score transform."""
import numpy as np
from morie.fn.sgnsc import sgnsc


def test_sgnsc_smoke():
    Z = np.array([1, 5, 2, 8, 3, 7, 4, 6, 9, 10], dtype=float)
    r = sgnsc(Z)
    assert r.name == "normal_score_transform"
    assert "transformed" in r.extra
    assert len(r.extra["transformed"]) == 10
    assert abs(np.mean(r.extra["transformed"])) < 0.5


def test_cheatsheet():
    from morie.fn.sgnsc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
