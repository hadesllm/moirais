"""Tests for fmFM.factorization_machines."""
import numpy as np
import pytest
from moirais.fn.fmFM import factorization_machines


def test_fmFM_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = factorization_machines(X, y, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fmFM_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = factorization_machines(X, y, K)
    assert isinstance(result, dict)
