"""Tests for polkrn.polynomial_kernel_msm."""
import numpy as np
import pytest
from moirais.fn.polkrn import polynomial_kernel_msm


def test_polkrn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A_history = np.random.default_rng(42).normal(0, 1, 100)
    H_history = np.random.default_rng(42).normal(0, 1, 100)
    degree = np.random.default_rng(42).normal(0, 1, 100)
    result = polynomial_kernel_msm(y, A_history, H_history, degree)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_polkrn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A_history = np.random.default_rng(42).normal(0, 1, 100)
    H_history = np.random.default_rng(42).normal(0, 1, 100)
    degree = np.random.default_rng(42).normal(0, 1, 100)
    result = polynomial_kernel_msm(y, A_history, H_history, degree)
    assert isinstance(result, dict)
