"""Tests for moirais.fn.entrp — Metric entropy (covering number) computation."""

import numpy as np
import pytest

from moirais.fn.entrp import entrp, MetricEntropyResult


def test_returns_result_type():
    X = np.random.default_rng(42).standard_normal((100, 3))
    result = entrp(X, epsilon=0.1)
    assert isinstance(result, MetricEntropyResult)


def test_log_cover_positive():
    X = np.random.default_rng(42).standard_normal((50, 2))
    result = entrp(X, epsilon=0.1)
    assert result.log_covering_number > 0


def test_covering_number_positive():
    X = np.random.default_rng(42).standard_normal((50, 2))
    result = entrp(X, epsilon=0.5)
    assert result.covering_number >= 1


def test_smaller_epsilon_larger_cover():
    X = np.random.default_rng(42).standard_normal((50, 2))
    r_big = entrp(X, epsilon=1.0)
    r_small = entrp(X, epsilon=0.1)
    assert r_small.log_covering_number > r_big.log_covering_number


def test_higher_dim_larger_cover():
    rng = np.random.default_rng(42)
    X2 = rng.standard_normal((50, 2))
    X5 = rng.standard_normal((50, 5))
    r2 = entrp(X2, epsilon=0.5)
    r5 = entrp(X5, epsilon=0.5)
    assert r5.log_covering_number > r2.log_covering_number


def test_diameter_positive():
    X = np.random.default_rng(42).standard_normal((50, 2))
    result = entrp(X, epsilon=0.1)
    assert result.diameter > 0


def test_entropy_integral_positive():
    X = np.random.default_rng(42).standard_normal((50, 2))
    result = entrp(X, epsilon=0.1)
    assert result.entropy_integral > 0


def test_1d_input():
    x = np.random.default_rng(42).standard_normal(100)
    result = entrp(x, epsilon=0.2)
    assert result.dimension == 1


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        entrp(np.array([]))


def test_bad_epsilon_raises():
    with pytest.raises(ValueError, match="epsilon"):
        entrp(np.array([1.0, 2.0]), epsilon=0.0)
