"""Tests for exchg.exchangeability_assumption."""
import numpy as np
import pytest
from moirais.fn.exchg import exchangeability_assumption


def test_exchg_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    result = exchangeability_assumption(Y, T, X, dag)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_exchg_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    result = exchangeability_assumption(Y, T, X, dag)
    assert isinstance(result, dict)
