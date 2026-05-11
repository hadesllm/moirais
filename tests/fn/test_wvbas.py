"""Tests for wvbas.py - Wavelet basis."""
import numpy as np
from morie.fn.wvbas import wavelet_basis, wvbas


def test_basis_returns_result():
    result = wavelet_basis("db4")
    assert result.name == "wavelet_basis"
    assert "lo_d" in result.extra
    assert "hi_d" in result.extra


def test_basis_haar():
    result = wavelet_basis("haar")
    assert result.extra["length"] == 2


def test_basis_reconstruction_filters():
    result = wavelet_basis("db4")
    assert "lo_r" in result.extra
    assert "hi_r" in result.extra


def test_basis_invalid():
    import pytest
    with pytest.raises(ValueError):
        wavelet_basis("unsupported")


def test_basis_alias():
    result = wvbas("db2")
    assert result.name == "wavelet_basis"
