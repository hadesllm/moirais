"""Tests for wvmra.py - Wavelet multiresolution analysis."""
import numpy as np
from moirais.fn.wvmra import wavelet_mra, wvmra


def test_wvmra_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(128)
    result = wavelet_mra(x, wavelet="haar", level=2)
    assert result.name == "wavelet_mra"
    assert "details" in result.extra
    assert "approximation" in result.extra


def test_wvmra_detail_count():
    x = np.random.default_rng(42).standard_normal(128)
    result = wavelet_mra(x, wavelet="haar", level=3)
    assert len(result.extra["details"]) == 3


def test_wvmra_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = wvmra(x, wavelet="haar", level=1)
    assert result.name == "wavelet_mra"
