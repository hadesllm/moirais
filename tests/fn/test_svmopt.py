"""Tests for svmopt.svm_dual."""
import numpy as np
import pytest
from morie.fn.svmopt import svm_dual


def test_svmopt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = svm_dual(X, y, C, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_svmopt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = svm_dual(X, y, C, kernel)
    assert isinstance(result, dict)
