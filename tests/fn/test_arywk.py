"""Tests for arywk.py - AR Yule-Walker estimation."""
import numpy as np
from moirais.fn.arywk import ar_yule_walker_fn, arywk


def test_arywk_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_yule_walker_fn(x, order=4)
    assert result.name == "ar_yule_walker"
    assert "coefficients" in result.extra
    assert len(result.extra["coefficients"]) == 4


def test_arywk_sigma2_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_yule_walker_fn(x)
    assert result.extra["sigma2"] > 0


def test_arywk_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = arywk(x, order=2)
    assert result.name == "ar_yule_walker"
    assert len(result.extra["coefficients"]) == 2
