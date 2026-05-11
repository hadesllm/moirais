"""Tests for eslols.esl_ols_normal_equations."""
import numpy as np
import pytest
from morie.fn.eslols import esl_ols_normal_equations


def test_eslols_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_ols_normal_equations(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslols_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_ols_normal_equations(X, y)
    assert isinstance(result, dict)
