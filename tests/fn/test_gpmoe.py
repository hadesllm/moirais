"""Tests for gpmoe.gp_mixture_of_experts."""
import numpy as np
import pytest
from morie.fn.gpmoe import gp_mixture_of_experts


def test_gpmoe_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = gp_mixture_of_experts(X, y, X_test, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpmoe_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = gp_mixture_of_experts(X, y, X_test, K)
    assert isinstance(result, dict)
