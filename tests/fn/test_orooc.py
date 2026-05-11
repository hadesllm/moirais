"""Tests for ordered optimal classification."""
import numpy as np
from morie.fn.orooc import orooc


def test_orooc_smoke():
    rng = np.random.default_rng(42)
    Y = rng.integers(1, 5, size=(20, 5)).astype(float)
    r = orooc(Y)
    assert r.name == "ordered_oc"
    assert "ideal_points" in r.extra
    assert "cutpoints" in r.extra


def test_cheatsheet():
    from morie.fn.orooc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
