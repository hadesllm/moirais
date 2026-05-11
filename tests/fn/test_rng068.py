"""Tests for rng068.rangayyan_ch3_dft_K_samples."""
import numpy as np
import pytest
from morie.fn.rng068 import rangayyan_ch3_dft_K_samples


def test_rng068_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    k = 5
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    N = 100
    result = rangayyan_ch3_dft_K_samples(x, n, k, K, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng068_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    k = 5
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    N = 100
    result = rangayyan_ch3_dft_K_samples(x, n, k, K, N)
    assert isinstance(result, dict)
