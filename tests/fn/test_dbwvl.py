"""Tests for dbwvl.py - Daubechies wavelet."""
import numpy as np
from moirais.fn.dbwvl import daubechies_wavelet, dbwvl


def test_db4_returns_result():
    result = daubechies_wavelet(4)
    assert result.name == "daubechies_wavelet"
    assert result.extra["length"] == 8


def test_db1_haar():
    result = daubechies_wavelet(1)
    lo = result.extra["lo_d"]
    assert len(lo) == 2
    assert abs(np.sum(lo ** 2) - 1.0) < 0.01


def test_db_filters_orthogonal():
    result = daubechies_wavelet(4)
    lo = result.extra["lo_d"]
    hi = result.extra["hi_d"]
    dot = np.dot(lo, hi)
    assert abs(dot) < 0.01


def test_db_invalid_order():
    import pytest
    with pytest.raises(ValueError):
        daubechies_wavelet(99)


def test_db_alias():
    result = dbwvl(2)
    assert result.name == "daubechies_wavelet"
