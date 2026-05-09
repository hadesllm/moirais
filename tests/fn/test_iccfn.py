"""Tests for moirais.fn.iccfn -- ICC curve data."""
import numpy as np
from moirais.fn.iccfn import icc_curve_data, iccfn


def test_alias():
    assert iccfn is icc_curve_data


def test_smoke():
    r = icc_curve_data(alpha=1.5, beta=0.0)
    assert r.name == "icc_curve_data"
    assert len(r.extra["theta"]) == 200
    assert len(r.extra["probability"]) == 200
    mid_idx = 100
    assert abs(r.extra["probability"][mid_idx] - 0.5) < 0.1


def test_custom_range():
    r = icc_curve_data(alpha=1.0, beta=0.0, theta_range=(-1, 1))
    assert r.extra["theta"][0] >= -1.0
    assert r.extra["theta"][-1] <= 1.0
