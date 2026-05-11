"""Tests for morie.fn.brkts — Bracketing number estimation."""

import numpy as np
import pytest

from morie.fn.brkts import brkts, BracketingResult


def test_returns_result_type():
    X = np.random.default_rng(42).standard_normal((100, 3))
    result = brkts(X, epsilon=0.1)
    assert isinstance(result, BracketingResult)


def test_log_bracket_positive():
    X = np.random.default_rng(42).standard_normal((50, 2))
    result = brkts(X, epsilon=0.1)
    assert result.log_bracketing_number > 0


def test_n_brackets_positive():
    X = np.random.default_rng(42).standard_normal((50, 2))
    result = brkts(X, epsilon=0.1)
    assert result.n_brackets >= 1


def test_smaller_epsilon_larger_brackets():
    X = np.random.default_rng(42).standard_normal((50, 2))
    r_big = brkts(X, epsilon=0.5)
    r_small = brkts(X, epsilon=0.1)
    assert r_small.log_bracketing_number > r_big.log_bracketing_number


def test_higher_dim_larger_brackets():
    rng = np.random.default_rng(42)
    X2 = rng.standard_normal((50, 2))
    X5 = rng.standard_normal((50, 5))
    r2 = brkts(X2, epsilon=0.1)
    r5 = brkts(X5, epsilon=0.1)
    assert r5.log_bracketing_number > r2.log_bracketing_number


def test_entropy_integral_positive():
    X = np.random.default_rng(42).standard_normal((50, 2))
    result = brkts(X, epsilon=0.1)
    assert result.entropy_integral > 0


def test_dimension_stored():
    X = np.random.default_rng(42).standard_normal((50, 4))
    result = brkts(X, epsilon=0.1)
    assert result.dimension == 4


def test_1d_input():
    x = np.random.default_rng(42).standard_normal(100)
    result = brkts(x, epsilon=0.2)
    assert result.dimension == 1


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        brkts(np.array([]))


def test_bad_epsilon_raises():
    with pytest.raises(ValueError, match="epsilon"):
        brkts(np.array([1.0, 2.0]), epsilon=-0.1)
