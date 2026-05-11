"""Tests for morie.fn.gauu — gaussian utility."""
import numpy as np
from morie.fn.gauu import gauu


def test_gauu_smoke():
    r = gauu(0.0, 1.0)
    assert r.name == "gaussian_utility"
    assert 0 < r.value < 1
    assert "w" in r.extra


def test_gauu_zero_distance():
    r = gauu(2.0, 2.0)
    assert abs(r.value - 1.0) < 1e-10
