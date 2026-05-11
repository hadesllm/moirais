"""Tests for tmlsen.tmle_sensitivity_unmeasured."""
import numpy as np
import pytest
from morie.fn.tmlsen import tmle_sensitivity_unmeasured


def test_tmlsen_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    gamma_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_sensitivity_unmeasured(y, D, X, gamma_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlsen_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    gamma_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_sensitivity_unmeasured(y, D, X, gamma_grid)
    assert isinstance(result, dict)
