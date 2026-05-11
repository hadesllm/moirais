"""Tests for tqkmse.turboquant_kv_mse."""
import numpy as np
import pytest
from morie.fn.tqkmse import turboquant_kv_mse


def test_tqkmse_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = turboquant_kv_mse(x, bits, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqkmse_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = turboquant_kv_mse(x, bits, method)
    assert isinstance(result, dict)
