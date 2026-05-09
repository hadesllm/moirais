"""Tests for svmsl.svm_soft_margin."""
import numpy as np
import pytest
from moirais.fn.svmsl import svm_soft_margin


def test_svmsl_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = svm_soft_margin(X, y, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_svmsl_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = svm_soft_margin(X, y, C)
    assert isinstance(result, dict)
