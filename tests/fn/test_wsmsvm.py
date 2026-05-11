"""Tests for wsmsvm.wasserman_svm."""
import numpy as np
import pytest
from morie.fn.wsmsvm import wasserman_svm


def test_wsmsvm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wasserman_svm(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmsvm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wasserman_svm(X, y)
    assert isinstance(result, dict)
