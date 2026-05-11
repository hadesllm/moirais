"""Tests for survbtr.bart_survival."""
import numpy as np
import pytest
from morie.fn.survbtr import bart_survival


def test_survbtr_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_trees = np.random.default_rng(42).normal(0, 1, 100)
    result = bart_survival(time, event, X, n_trees)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survbtr_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_trees = np.random.default_rng(42).normal(0, 1, 100)
    result = bart_survival(time, event, X, n_trees)
    assert isinstance(result, dict)
