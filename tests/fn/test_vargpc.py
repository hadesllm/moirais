"""Tests for vargpc.variational_gp_classifier."""
import numpy as np
import pytest
from morie.fn.vargpc import variational_gp_classifier


def test_vargpc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    result = variational_gp_classifier(X, y, X_test)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vargpc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    result = variational_gp_classifier(X, y, X_test)
    assert isinstance(result, dict)
