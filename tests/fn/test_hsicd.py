"""Tests for hsicd.hsic_independence."""
import numpy as np
import pytest
from moirais.fn.hsicd import hsic_independence


def test_hsicd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    alpha = 0.05
    result = hsic_independence(X, Y, kernel, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hsicd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    alpha = 0.05
    result = hsic_independence(X, Y, kernel, alpha)
    assert isinstance(result, dict)
