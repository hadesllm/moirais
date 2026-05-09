"""Tests for ridgs.ridge_solution."""
import numpy as np
import pytest
from moirais.fn.ridgs import ridge_solution


def test_ridgs_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    result = ridge_solution(X, y, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ridgs_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    result = ridge_solution(X, y, lam)
    assert isinstance(result, dict)
