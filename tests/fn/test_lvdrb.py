"""Tests for lvdrb.py - Levinson-Durbin recursion."""
import numpy as np
from moirais.fn.lvdrb import levinson_durbin_fn, lvdrb


def test_lvdrb_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    r = np.correlate(x, x, mode="full")
    n = len(x)
    acf = r[n - 1:n + 5] / n
    result = levinson_durbin_fn(acf, order=4)
    assert result.name == "levinson_durbin"
    assert len(result.extra["coefficients"]) == 4


def test_lvdrb_sigma2_positive():
    x = np.random.default_rng(42).standard_normal(256)
    r = np.correlate(x, x, mode="full")
    n = len(x)
    acf = r[n - 1:n + 5] / n
    result = levinson_durbin_fn(acf, order=4)
    assert result.extra["sigma2"] > 0


def test_lvdrb_alias():
    acf = np.array([1.0, 0.5, 0.2, 0.1])
    result = lvdrb(acf, order=2)
    assert result.name == "levinson_durbin"
