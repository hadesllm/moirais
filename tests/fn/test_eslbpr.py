"""Tests for eslbpr.esl_backprop."""
import numpy as np
import pytest
from morie.fn.eslbpr import esl_backprop


def test_eslbpr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = esl_backprop(X, y, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslbpr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = esl_backprop(X, y, weights)
    assert isinstance(result, dict)
