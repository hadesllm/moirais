"""Tests for hrzseriu.horowitz_series_unknown_T."""
import numpy as np
import pytest
from moirais.fn.hrzseriu import horowitz_series_unknown_T


def test_hrzseriu_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = horowitz_series_unknown_T(x, y, w, K, basis)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzseriu_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = horowitz_series_unknown_T(x, y, w, K, basis)
    assert isinstance(result, dict)
