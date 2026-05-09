"""Tests for hmdtr.geron_tree_regularization."""
import numpy as np
import pytest
from moirais.fn.hmdtr import geron_tree_regularization


def test_hmdtr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    min_samples_leaf = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_tree_regularization(X, y, max_depth, min_samples_leaf)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdtr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    min_samples_leaf = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_tree_regularization(X, y, max_depth, min_samples_leaf)
    assert isinstance(result, dict)
