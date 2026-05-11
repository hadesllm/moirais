"""Tests for hmhgb.geron_histogram_gradient_boosting."""
import numpy as np
import pytest
from morie.fn.hmhgb import geron_histogram_gradient_boosting


def test_hmhgb_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    learning_rate = 0.1
    max_bins = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_histogram_gradient_boosting(X, y, max_iter, learning_rate, max_bins)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmhgb_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    learning_rate = 0.1
    max_bins = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_histogram_gradient_boosting(X, y, max_iter, learning_rate, max_bins)
    assert isinstance(result, dict)
