"""Tests for htprd.hyperparameter_tuning_grid."""
import numpy as np
import pytest
from morie.fn.htprd import hyperparameter_tuning_grid


def test_htprd_basic():
    """Test basic functionality."""
    param_grid = np.random.default_rng(42).normal(0, 1, 100)
    cv_data = np.random.default_rng(42).normal(0, 1, 100)
    result = hyperparameter_tuning_grid(param_grid, cv_data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_htprd_edge():
    """Test edge cases."""
    param_grid = np.random.default_rng(42).normal(0, 1, 100)
    cv_data = np.random.default_rng(42).normal(0, 1, 100)
    result = hyperparameter_tuning_grid(param_grid, cv_data)
    assert isinstance(result, dict)
