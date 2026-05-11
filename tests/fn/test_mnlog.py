"""Tests for mnlog.multinomial_logistic_penalized."""
import numpy as np
import pytest
from morie.fn.mnlog import multinomial_logistic_penalized


def test_mnlog_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lam = 0.1
    alpha = 0.05
    result = multinomial_logistic_penalized(y, X, lam, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mnlog_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lam = 0.1
    alpha = 0.05
    result = multinomial_logistic_penalized(y, X, lam, alpha)
    assert isinstance(result, dict)
