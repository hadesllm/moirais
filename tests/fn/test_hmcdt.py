"""Tests for hmcdt.geron_classification_tree."""
import numpy as np
import pytest
from moirais.fn.hmcdt import geron_classification_tree


def test_hmcdt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    criterion = np.random.default_rng(42).normal(0, 1, 100)
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_classification_tree(X, y, criterion, max_depth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmcdt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    criterion = np.random.default_rng(42).normal(0, 1, 100)
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_classification_tree(X, y, criterion, max_depth)
    assert isinstance(result, dict)
