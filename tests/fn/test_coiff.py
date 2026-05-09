"""Tests for coiff.py - Coiflet wavelet filter coefficients."""
import numpy as np
from moirais.fn.coiff import coiflet_coeffs, coiff


def test_coiff_returns_descriptive_result():
    result = coiflet_coeffs(order=1)
    assert result.name == "coiflet_coeffs"
    assert "lo_d" in result.extra


def test_coiff_filter_length():
    result = coiflet_coeffs(order=2)
    assert len(result.extra["lo_d"]) == 12


def test_coiff_alias():
    result = coiff(order=1)
    assert result.name == "coiflet_coeffs"
