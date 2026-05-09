"""Tests for bnscbo.bound_compound_outcome."""
import numpy as np
import pytest
from moirais.fn.bnscbo import bound_compound_outcome


def test_bnscbo_basic():
    """Test basic functionality."""
    y_components = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = bound_compound_outcome(y_components, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnscbo_edge():
    """Test edge cases."""
    y_components = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = bound_compound_outcome(y_components, D, X)
    assert isinstance(result, dict)
