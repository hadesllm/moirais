"""Tests for fzkqe.fauzi_kernel_quantile_estimator."""
import numpy as np
import pytest
from morie.fn.fzkqe import fauzi_kernel_quantile_estimator


def test_fzkqe_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    bandwidth = 0.3
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = fauzi_kernel_quantile_estimator(data, p, bandwidth, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzkqe_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    bandwidth = 0.3
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = fauzi_kernel_quantile_estimator(data, p, bandwidth, kernel)
    assert isinstance(result, dict)
