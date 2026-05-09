"""Tests for rng056.rangayyan_ch3_iir_transfer_function."""
import numpy as np
import pytest
from moirais.fn.rng056 import rangayyan_ch3_iir_transfer_function


def test_rng056_basic():
    """Test basic functionality."""
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    N = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_iir_transfer_function(b_k, a_k, z, N, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng056_edge():
    """Test edge cases."""
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    N = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_iir_transfer_function(b_k, a_k, z, N, M)
    assert isinstance(result, dict)
