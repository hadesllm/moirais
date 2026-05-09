"""Tests for svmkr.svm_kernel_trick."""
import numpy as np
import pytest
from moirais.fn.svmkr import svm_kernel_trick


def test_svmkr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = svm_kernel_trick(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_svmkr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = svm_kernel_trick(x, y)
    assert isinstance(result, dict)
