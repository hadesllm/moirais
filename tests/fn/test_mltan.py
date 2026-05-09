"""Tests for mltan.py - Multiresolution analysis."""
import numpy as np
from moirais.fn.mltan import multiresolution_analysis, mltan


def test_mra_returns_result():
    x = np.random.default_rng(42).standard_normal(128)
    result = multiresolution_analysis(x, level=3)
    assert result.name == "multiresolution_analysis"
    assert "approximation" in result.extra
    assert "details" in result.extra


def test_mra_detail_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = multiresolution_analysis(x, level=3)
    assert len(result.extra["details"]) == 3


def test_mra_haar():
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    result = multiresolution_analysis(x, wavelet="haar", level=2)
    assert result.extra["level"] == 2


def test_mra_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = mltan(x, level=2)
    assert result.name == "multiresolution_analysis"
