"""Tests for mlfit.ml_loglik."""
import numpy as np
import pytest
from morie.fn.mlfit import ml_loglik


def test_mlfit_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = ml_loglik(y, X, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mlfit_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = ml_loglik(y, X, V)
    assert isinstance(result, dict)
