"""Tests for morie.fn.occln -- OC cutting line."""
import numpy as np
from morie.fn.occln import oc_cutting_line, occln


def test_alias():
    assert occln is oc_cutting_line


def test_smoke():
    r = oc_cutting_line(normal=[1.0, 1.0], cutpoint=0.5)
    assert r.name == "oc_cutting_line"
    assert "slope" in r.extra
    assert "intercept" in r.extra


def test_known_line():
    r = oc_cutting_line(normal=[0.0, 1.0], cutpoint=0.5)
    assert abs(r.extra["slope"]) < 1e-10
    assert abs(r.extra["intercept"] - 0.5) < 1e-10
