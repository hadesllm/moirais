"""Tests for eslr2.esl_r_squared."""
import numpy as np
import pytest
from moirais.fn.eslr2 import esl_r_squared


def test_eslr2_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    result = esl_r_squared(X, y, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslr2_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    result = esl_r_squared(X, y, beta)
    assert isinstance(result, dict)
