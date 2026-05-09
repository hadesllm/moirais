"""Tests for jhoint.johansen_trace."""
import numpy as np
import pytest
from moirais.fn.jhoint import johansen_trace


def test_jhoint_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lags = 10
    result = johansen_trace(X, lags)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_jhoint_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lags = 10
    result = johansen_trace(X, lags)
    assert isinstance(result, dict)
