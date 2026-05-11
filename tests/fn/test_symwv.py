"""Tests for symwv.py - Symlet wavelet."""
import numpy as np
from morie.fn.symwv import symlet_wavelet, symwv


def test_sym4_returns_result():
    result = symlet_wavelet(4)
    assert result.name == "symlet_wavelet"
    assert result.extra["length"] == 8


def test_sym_has_filters():
    result = symlet_wavelet(3)
    assert "lo_d" in result.extra
    assert "hi_d" in result.extra


def test_sym_invalid_order():
    import pytest
    with pytest.raises(ValueError):
        symlet_wavelet(99)


def test_sym_alias():
    result = symwv(2)
    assert result.name == "symlet_wavelet"
