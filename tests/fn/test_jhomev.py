"""Tests for jhomev.johansen_max_eigen."""
import numpy as np
import pytest
from moirais.fn.jhomev import johansen_max_eigen


def test_jhomev_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lags = 10
    result = johansen_max_eigen(X, lags)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_jhomev_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lags = 10
    result = johansen_max_eigen(X, lags)
    assert isinstance(result, dict)
