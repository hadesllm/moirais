"""Tests for db4fn.py - Daubechies wavelet filter coefficients."""
import numpy as np
from moirais.fn.db4fn import daubechies_coeffs, db4fn


def test_db4fn_returns_descriptive_result():
    result = daubechies_coeffs(order=4)
    assert result.name == "daubechies_coeffs"
    assert "lo_d" in result.extra


def test_db4fn_filter_length():
    result = daubechies_coeffs(order=2)
    assert len(result.extra["lo_d"]) == 4


def test_db4fn_alias():
    result = db4fn(order=1)
    assert result.name == "daubechies_coeffs"
