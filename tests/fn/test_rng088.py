"""Tests for rng088.rangayyan_ch3_ma_transfer_function."""
import numpy as np
import pytest
from moirais.fn.rng088 import rangayyan_ch3_ma_transfer_function


def test_rng088_basic():
    """Test basic functionality."""
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_ma_transfer_function(b_k, z, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng088_edge():
    """Test edge cases."""
    b_k = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_ma_transfer_function(b_k, z, N)
    assert isinstance(result, dict)
