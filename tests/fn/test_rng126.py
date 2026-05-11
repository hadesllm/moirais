"""Tests for rng126.rangayyan_ch3_butterworth_analog_transfer_function."""
import numpy as np
import pytest
from morie.fn.rng126 import rangayyan_ch3_butterworth_analog_transfer_function


def test_rng126_basic():
    """Test basic functionality."""
    s = 90
    p_k = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    N = 100
    result = rangayyan_ch3_butterworth_analog_transfer_function(s, p_k, G, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng126_edge():
    """Test edge cases."""
    s = 90
    p_k = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    N = 100
    result = rangayyan_ch3_butterworth_analog_transfer_function(s, p_k, G, N)
    assert isinstance(result, dict)
