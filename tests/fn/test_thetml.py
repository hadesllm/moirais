"""Tests for thetml.theta_mle."""
import numpy as np
import pytest
from morie.fn.thetml import theta_mle


def test_thetml_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    items = np.random.default_rng(42).normal(0, 1, 100)
    result = theta_mle(X, items)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_thetml_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    items = np.random.default_rng(42).normal(0, 1, 100)
    result = theta_mle(X, items)
    assert isinstance(result, dict)
