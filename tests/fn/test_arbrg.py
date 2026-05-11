"""Tests for arbrg.py - AR Burg estimation."""
import numpy as np
import pytest
from morie.fn.arbrg import ar_burg_fn, arbrg


def test_ar_burg_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_burg_fn(x)
    assert result.name == "ar_burg"
    assert "coefficients" in result.extra
    assert "sigma2" in result.extra


def test_ar_burg_coefficient_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_burg_fn(x, order=8)
    assert len(result.extra["coefficients"]) == 8


def test_ar_burg_sigma2_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_burg_fn(x)
    assert result.extra["sigma2"] > 0


def test_arbrg_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = arbrg(x, order=5)
    assert result.name == "ar_burg"
