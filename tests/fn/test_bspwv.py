"""Tests for bspwv.py - B-spline wavelet."""
import numpy as np
from morie.fn.bspwv import bspline_wavelet, bspwv


def test_bspline_returns_result():
    result = bspline_wavelet(order=3)
    assert result.name == "bspline_wavelet"
    assert "wavelet" in result.extra


def test_bspline_normalized():
    result = bspline_wavelet(order=3)
    psi = result.extra["wavelet"]
    energy = np.sum(psi ** 2)
    assert abs(energy - 1.0) < 0.1


def test_bspline_order1():
    result = bspline_wavelet(order=1)
    assert result.extra["order"] == 1


def test_bspline_alias():
    result = bspwv(order=2)
    assert result.name == "bspline_wavelet"
