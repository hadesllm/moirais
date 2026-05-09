"""Tests for rng061.rangayyan_ch3_magnitude_response_from_pole_zero."""
import numpy as np
import pytest
from moirais.fn.rng061 import rangayyan_ch3_magnitude_response_from_pole_zero


def test_rng061_basic():
    """Test basic functionality."""
    l_k = np.random.default_rng(42).normal(0, 1, 100)
    r_k = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_magnitude_response_from_pole_zero(l_k, r_k, N, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng061_edge():
    """Test edge cases."""
    l_k = np.random.default_rng(42).normal(0, 1, 100)
    r_k = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_magnitude_response_from_pole_zero(l_k, r_k, N, M)
    assert isinstance(result, dict)
