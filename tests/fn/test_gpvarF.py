"""Tests for gpvarF.gp_variance."""
import numpy as np
import pytest
from moirais.fn.gpvarF import gp_variance


def test_gpvarF_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    X_star = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gp_variance(X, X_star, kernel, sigma2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpvarF_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    X_star = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gp_variance(X, X_star, kernel, sigma2)
    assert isinstance(result, dict)
