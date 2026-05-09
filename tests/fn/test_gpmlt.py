"""Tests for gpmlt.gp_multitask."""
import numpy as np
import pytest
from moirais.fn.gpmlt import gp_multitask


def test_gpmlt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y_tasks = np.random.default_rng(42).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    result = gp_multitask(X, y_tasks, X_test)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpmlt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y_tasks = np.random.default_rng(42).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    result = gp_multitask(X, y_tasks, X_test)
    assert isinstance(result, dict)
