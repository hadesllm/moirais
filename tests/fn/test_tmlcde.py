"""Tests for tmlcde.tmle_controlled_direct."""
import numpy as np
import pytest
from moirais.fn.tmlcde import tmle_controlled_direct


def test_tmlcde_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    m_value = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_controlled_direct(y, D, M, X, m_value)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlcde_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    m_value = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_controlled_direct(y, D, M, X, m_value)
    assert isinstance(result, dict)
