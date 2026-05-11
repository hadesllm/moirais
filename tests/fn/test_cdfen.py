"""Tests for cdf_estimate."""
import numpy as np
from morie.fn.cdfen import cdf_estimate, cdfen


def test_basic():
    r = cdf_estimate([3, 1, 2])
    assert r.extra["ecdf"][-1] == 1.0
    assert r.extra["x_sorted"] == [1.0, 2.0, 3.0]


def test_alias():
    assert cdfen is cdf_estimate


def test_single():
    r = cdf_estimate([5])
    assert r.extra["ecdf"] == [1.0]
