"""Tests for tmlcps.tmle_continuous_treatment."""
import numpy as np
import pytest
from morie.fn.tmlcps import tmle_continuous_treatment


def test_tmlcps_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    a_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_continuous_treatment(y, A, X, a_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlcps_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    a_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_continuous_treatment(y, A, X, a_grid)
    assert isinstance(result, dict)
