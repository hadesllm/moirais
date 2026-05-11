"""Tests for prdny.py - Prony's method."""
import numpy as np
from morie.fn.prdny import prony_method_fn, prdny


def test_prdny_returns_result():
    x = np.random.default_rng(42).standard_normal(64)
    result = prony_method_fn(x, p=3, q=2)
    assert result.name == "prony_method"
    assert "ar_coeffs" in result.extra
    assert "poles" in result.extra


def test_prdny_coefficient_lengths():
    x = np.random.default_rng(42).standard_normal(64)
    result = prony_method_fn(x, p=4, q=2)
    assert len(result.extra["ar_coeffs"]) == 4
    assert len(result.extra["ma_coeffs"]) == 3


def test_prdny_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = prdny(x, p=2, q=1)
    assert result.name == "prony_method"
