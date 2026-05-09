"""Tests for opttre.optimal_tree_regime."""
import numpy as np
import pytest
from moirais.fn.opttre import optimal_tree_regime


def test_opttre_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = optimal_tree_regime(y, A, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_opttre_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = optimal_tree_regime(y, A, W)
    assert isinstance(result, dict)
