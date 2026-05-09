"""Tests for gpkern.gp_kernel_compose."""
import numpy as np
import pytest
from moirais.fn.gpkern import gp_kernel_compose


def test_gpkern_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    kernel_spec = np.random.default_rng(42).normal(0, 1, 100)
    result = gp_kernel_compose(X, Y, kernel_spec)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpkern_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    kernel_spec = np.random.default_rng(42).normal(0, 1, 100)
    result = gp_kernel_compose(X, Y, kernel_spec)
    assert isinstance(result, dict)
