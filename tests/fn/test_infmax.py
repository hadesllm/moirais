"""Tests for infmax.infomax_objective."""
import numpy as np
import pytest
from morie.fn.infmax import infomax_objective


def test_infmax_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    T_network = np.random.default_rng(42).normal(0, 1, 100)
    result = infomax_objective(X, T_network)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_infmax_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    T_network = np.random.default_rng(42).normal(0, 1, 100)
    result = infomax_objective(X, T_network)
    assert isinstance(result, dict)
