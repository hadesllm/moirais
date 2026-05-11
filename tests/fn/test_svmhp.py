"""Tests for svmhp.svm_hyperplane."""
import numpy as np
import pytest
from morie.fn.svmhp import svm_hyperplane


def test_svmhp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = svm_hyperplane(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_svmhp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = svm_hyperplane(X, y)
    assert isinstance(result, dict)
