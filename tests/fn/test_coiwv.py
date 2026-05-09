"""Tests for coiwv.py - Coiflet wavelet."""
import numpy as np
from moirais.fn.coiwv import coiflet_wavelet, coiwv


def test_coif1_returns_result():
    result = coiflet_wavelet(1)
    assert result.name == "coiflet_wavelet"
    assert result.extra["length"] == 6


def test_coif_has_filters():
    result = coiflet_wavelet(2)
    assert "lo_d" in result.extra
    assert "hi_d" in result.extra


def test_coif_invalid_order():
    import pytest
    with pytest.raises(ValueError):
        coiflet_wavelet(99)


def test_coif_alias():
    result = coiwv(1)
    assert result.name == "coiflet_wavelet"
