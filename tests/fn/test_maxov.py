"""Tests for maxov.py - Maximal Overlap DWT."""
import numpy as np
from morie.fn.maxov import maximal_overlap_dwt, maxov


def test_modwt_returns_result():
    x = np.random.default_rng(42).standard_normal(128)
    result = maximal_overlap_dwt(x, level=3)
    assert result.name == "maximal_overlap_dwt"
    assert "coeffs" in result.extra


def test_modwt_same_length():
    x = np.random.default_rng(42).standard_normal(256)
    result = maximal_overlap_dwt(x, level=3)
    for c in result.extra["coeffs"]:
        assert len(c) == len(x)


def test_modwt_coeffs_count():
    x = np.random.default_rng(42).standard_normal(128)
    result = maximal_overlap_dwt(x, level=3)
    assert len(result.extra["coeffs"]) == 4


def test_modwt_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = maxov(x, level=2)
    assert result.name == "maximal_overlap_dwt"
