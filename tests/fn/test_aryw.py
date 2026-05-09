"""Tests for aryw.py - AR Yule-Walker estimation."""
import numpy as np
import pytest
from moirais.fn.aryw import ar_yule_walker_fn, aryw


def test_ar_yule_walker_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_yule_walker_fn(x)
    assert result.name == "ar_yule_walker"
    assert "coefficients" in result.extra
    assert "sigma2" in result.extra


def test_ar_yule_walker_coefficient_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_yule_walker_fn(x, order=10)
    assert len(result.extra["coefficients"]) == 10


def test_ar_yule_walker_sigma2_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_yule_walker_fn(x)
    assert result.extra["sigma2"] > 0


def test_aryw_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = aryw(x, order=5)
    assert result.name == "ar_yule_walker"
