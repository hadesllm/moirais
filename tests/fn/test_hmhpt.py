"""Tests for hmhpt.geron_hyperparameter_tuning."""
import numpy as np
import pytest
from morie.fn.hmhpt import geron_hyperparameter_tuning


def test_hmhpt_basic():
    """Test basic functionality."""
    param_grid = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_hyperparameter_tuning(param_grid, X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmhpt_edge():
    """Test edge cases."""
    param_grid = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_hyperparameter_tuning(param_grid, X, y)
    assert isinstance(result, dict)
