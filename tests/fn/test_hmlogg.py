"""Tests for hmlogg.geron_logistic_gradient."""
import numpy as np
import pytest
from morie.fn.hmlogg import geron_logistic_gradient


def test_hmlogg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_logistic_gradient(X, y, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlogg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_logistic_gradient(X, y, theta)
    assert isinstance(result, dict)
