"""Tests for survrsf.random_survival_forest."""
import numpy as np
import pytest
from morie.fn.survrsf import random_survival_forest


def test_survrsf_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_trees = np.random.default_rng(42).normal(0, 1, 100)
    result = random_survival_forest(time, event, X, n_trees)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survrsf_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_trees = np.random.default_rng(42).normal(0, 1, 100)
    result = random_survival_forest(time, event, X, n_trees)
    assert isinstance(result, dict)
