"""Tests for relative structured variability."""
from morie.fn.sgrsv import sgrsv


def test_sgrsv_strong():
    r = sgrsv(0.1, 1.0)
    assert r.name == "relative_structured_variability"
    assert r.extra["RSV"] == 0.9
    assert r.extra["spatial_strength"] == "strong"


def test_sgrsv_weak():
    r = sgrsv(0.8, 1.0)
    assert abs(r.extra["RSV"] - 0.2) < 1e-10
    assert r.extra["spatial_strength"] == "very_weak"
