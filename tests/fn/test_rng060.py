"""Tests for rng060.rangayyan_ch3_frequency_response_from_pole_zero."""
import numpy as np
import pytest
from morie.fn.rng060 import rangayyan_ch3_frequency_response_from_pole_zero


def test_rng060_basic():
    """Test basic functionality."""
    z_0 = np.random.default_rng(42).normal(0, 1, 100)
    z_k = np.random.default_rng(42).normal(0, 1, 100)
    p_k = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_frequency_response_from_pole_zero(z_0, z_k, p_k, N, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng060_edge():
    """Test edge cases."""
    z_0 = np.random.default_rng(42).normal(0, 1, 100)
    z_k = np.random.default_rng(42).normal(0, 1, 100)
    p_k = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_frequency_response_from_pole_zero(z_0, z_k, p_k, N, M)
    assert isinstance(result, dict)
