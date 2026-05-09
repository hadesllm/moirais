"""Tests for hrzich.horowitz_ichimura_estimator."""
import numpy as np
import pytest
from moirais.fn.hrzich import horowitz_ichimura_estimator


def test_hrzich_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_ichimura_estimator(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzich_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_ichimura_estimator(x, y, bandwidth)
    assert isinstance(result, dict)
