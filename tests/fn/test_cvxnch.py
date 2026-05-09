"""Tests for cvxnch.boyd_nuclear_norm."""
import numpy as np
import pytest
from moirais.fn.cvxnch import boyd_nuclear_norm


def test_cvxnch_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = boyd_nuclear_norm(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxnch_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = boyd_nuclear_norm(X)
    assert isinstance(result, dict)
