"""Tests for sym8f.py - Symlet wavelet filter coefficients."""
import numpy as np
from morie.fn.sym8f import symlet_coeffs, sym8f


def test_sym8f_returns_descriptive_result():
    result = symlet_coeffs(order=8)
    assert result.name == "symlet_coeffs"
    assert "lo_d" in result.extra


def test_sym8f_filter_length():
    result = symlet_coeffs(order=4)
    assert len(result.extra["lo_d"]) == 8


def test_sym8f_alias():
    result = sym8f(order=2)
    assert result.name == "symlet_coeffs"
