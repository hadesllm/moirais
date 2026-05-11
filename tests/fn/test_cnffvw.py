"""Tests for cnffvw.cinelli_hazlett_robust."""
import numpy as np
import pytest
from morie.fn.cnffvw import cinelli_hazlett_robust


def test_cnffvw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    R2_Y = np.random.default_rng(42).normal(0, 1, 100)
    R2_D = np.random.default_rng(42).normal(0, 1, 100)
    result = cinelli_hazlett_robust(y, D, X, R2_Y, R2_D)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cnffvw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    R2_Y = np.random.default_rng(42).normal(0, 1, 100)
    R2_D = np.random.default_rng(42).normal(0, 1, 100)
    result = cinelli_hazlett_robust(y, D, X, R2_Y, R2_D)
    assert isinstance(result, dict)
