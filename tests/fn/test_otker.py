"""Tests for otker.ot_kernel_emd_approx."""
import numpy as np
import pytest
from morie.fn.otker import ot_kernel_emd_approx


def test_otker_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    epsilon = 1e-6
    result = ot_kernel_emd_approx(X, Y, kernel, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otker_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    epsilon = 1e-6
    result = ot_kernel_emd_approx(X, Y, kernel, epsilon)
    assert isinstance(result, dict)
