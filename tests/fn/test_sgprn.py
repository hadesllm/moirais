"""Tests for practical range."""
import numpy as np
from morie.fn.sgprn import sgprn


def test_sgprn_spherical():
    r = sgprn("spherical", {"nugget": 0, "sill": 1, "range": 5.0})
    assert r.name == "practical_range"
    assert r.extra["practical_range"] == 5.0


def test_sgprn_exponential():
    r = sgprn("exponential", {"nugget": 0, "sill": 1, "range": 2.0})
    assert r.extra["practical_range"] > 2.0
