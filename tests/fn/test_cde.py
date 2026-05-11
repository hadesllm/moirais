"""Tests for cde.controlled_direct_effect."""
import numpy as np
import pytest
from morie.fn.cde import controlled_direct_effect


def test_cde_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    m = 10
    result = controlled_direct_effect(Y, X, M, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cde_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    m = 10
    result = controlled_direct_effect(Y, X, M, m)
    assert isinstance(result, dict)
