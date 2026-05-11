"""Tests for morie.fn.clvr — clever covariate."""

import numpy as np
import pytest

from morie.fn.clvr import clvr


def test_basic_output():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    result = clvr(T, X)
    assert "H" in result
    assert "H1" in result
    assert "H0" in result
    assert result["n"] == 200


def test_h1_positive():
    rng = np.random.default_rng(7)
    n = 100
    X = rng.standard_normal((n, 1))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    result = clvr(T, X)
    assert np.all(result["H1"] > 0)


def test_h0_negative():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 1))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    result = clvr(T, X)
    assert np.all(result["H0"] < 0)
