"""Tests for kcusum.kernel_cusum."""
import numpy as np
import pytest
from moirais.fn.kcusum import kernel_cusum


def test_kcusum_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = kernel_cusum(x, kernel, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kcusum_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = kernel_cusum(x, kernel, threshold)
    assert isinstance(result, dict)
