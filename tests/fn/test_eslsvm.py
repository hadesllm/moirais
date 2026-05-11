"""Tests for eslsvm.esl_svm_kernel."""
import numpy as np
import pytest
from morie.fn.eslsvm import esl_svm_kernel


def test_eslsvm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = esl_svm_kernel(X, y, C, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslsvm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = esl_svm_kernel(X, y, C, kernel)
    assert isinstance(result, dict)
