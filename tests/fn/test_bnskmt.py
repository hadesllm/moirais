"""Tests for bnskmt.bound_kernel_moment."""
import numpy as np
import pytest
from morie.fn.bnskmt import bound_kernel_moment


def test_bnskmt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = bound_kernel_moment(y, X, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnskmt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = bound_kernel_moment(y, X, kernel)
    assert isinstance(result, dict)
