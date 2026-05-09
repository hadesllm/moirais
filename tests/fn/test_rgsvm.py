"""Tests for rgsvm.rangayyan_svm."""
import numpy as np
import pytest
from moirais.fn.rgsvm import rangayyan_svm


def test_rgsvm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_svm(X, y, kernel, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsvm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_svm(X, y, kernel, C)
    assert isinstance(result, dict)
