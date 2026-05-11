"""Tests for varKf.variational_gp."""
import numpy as np
import pytest
from morie.fn.varKf import variational_gp


def test_varKf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = variational_gp(X, y, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_varKf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = variational_gp(X, y, Z)
    assert isinstance(result, dict)
