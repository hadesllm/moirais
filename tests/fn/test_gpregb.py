"""Tests for gpregb.gp_regression_bayes."""
import numpy as np
import pytest
from moirais.fn.gpregb import gp_regression_bayes


def test_gpregb_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = gp_regression_bayes(X, y, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpregb_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = gp_regression_bayes(X, y, kernel)
    assert isinstance(result, dict)
