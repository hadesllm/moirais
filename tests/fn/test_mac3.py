"""Tests for mac3.ma_centered_predictors."""
import numpy as np
import pytest
from morie.fn.mac3 import ma_centered_predictors


def test_mac3_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    result = ma_centered_predictors(X, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mac3_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    result = ma_centered_predictors(X, weights)
    assert isinstance(result, dict)
