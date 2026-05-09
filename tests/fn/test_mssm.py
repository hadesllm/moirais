"""Tests for mssm.marginal_structural_med."""
import numpy as np
import pytest
from moirais.fn.mssm import marginal_structural_med


def test_mssm_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = marginal_structural_med(Y, X, M, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mssm_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = marginal_structural_med(Y, X, M, C)
    assert isinstance(result, dict)
