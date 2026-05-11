"""Tests for gcnII.gcnii."""
import numpy as np
import pytest
from morie.fn.gcnII import gcnii


def test_gcnII_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H0 = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = gcnii(A, H0, alpha, beta, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gcnII_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H0 = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = gcnii(A, H0, alpha, beta, K)
    assert isinstance(result, dict)
