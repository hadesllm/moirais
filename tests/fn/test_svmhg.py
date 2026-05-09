"""Tests for svmhg.svm_hinge_primal."""
import numpy as np
import pytest
from moirais.fn.svmhg import svm_hinge_primal


def test_svmhg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = svm_hinge_primal(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_svmhg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = svm_hinge_primal(x, y)
    assert isinstance(result, dict)
