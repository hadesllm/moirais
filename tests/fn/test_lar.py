"""Tests for lar.py - Log Area Ratios."""
import numpy as np
from morie.fn.lar import log_area_ratio_fn, lar


def test_lar_returns_result():
    coeffs = np.array([0.5, -0.3, 0.1])
    result = log_area_ratio_fn(coeffs)
    assert result.name == "log_area_ratio"
    assert "lar" in result.extra
    assert "reflection_coeffs" in result.extra


def test_lar_finite():
    coeffs = np.array([0.3, -0.2])
    result = log_area_ratio_fn(coeffs)
    assert np.all(np.isfinite(result.extra["lar"]))


def test_lar_alias():
    coeffs = np.array([0.4, -0.1])
    result = lar(coeffs)
    assert result.name == "log_area_ratio"
