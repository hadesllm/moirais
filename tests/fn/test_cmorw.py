"""Tests for cmorw.py - Complex Morlet wavelet."""
import numpy as np
from morie.fn.cmorw import cmor_wavelet, cmorw


def test_cmorw_returns_descriptive_result():
    result = cmor_wavelet(fb=1.5, fc=1.0, N=256)
    assert result.name == "cmor_wavelet"
    assert "wavelet" in result.extra


def test_cmorw_complex():
    result = cmor_wavelet()
    assert np.iscomplexobj(result.extra["wavelet"])


def test_cmorw_alias():
    result = cmorw()
    assert result.name == "cmor_wavelet"
