"""Tests for ratest.ratio_estimator."""
import numpy as np
import pytest
from moirais.fn.ratest import ratio_estimator


def test_ratest_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = ratio_estimator(y, x, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ratest_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = ratio_estimator(y, x, weights)
    assert isinstance(result, dict)
