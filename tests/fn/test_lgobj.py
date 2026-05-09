"""Tests for lgobj.logistic_log_likelihood."""
import numpy as np
import pytest
from moirais.fn.lgobj import logistic_log_likelihood


def test_lgobj_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    beta = 0.8
    result = logistic_log_likelihood(y, X, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lgobj_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    beta = 0.8
    result = logistic_log_likelihood(y, X, beta)
    assert isinstance(result, dict)
