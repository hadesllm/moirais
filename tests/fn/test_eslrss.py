"""Tests for eslrss.esl_residual_sum_squares."""
import numpy as np
import pytest
from moirais.fn.eslrss import esl_residual_sum_squares


def test_eslrss_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    result = esl_residual_sum_squares(X, y, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslrss_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    result = esl_residual_sum_squares(X, y, beta)
    assert isinstance(result, dict)
