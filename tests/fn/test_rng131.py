"""Tests for rng131.rangayyan_ch3_butterworth_digital_transfer_function."""
import numpy as np
import pytest
from moirais.fn.rng131 import rangayyan_ch3_butterworth_digital_transfer_function


def test_rng131_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    G_prime = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_butterworth_digital_transfer_function(z, a_k, G_prime, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng131_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    a_k = np.random.default_rng(42).normal(0, 1, 100)
    G_prime = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_butterworth_digital_transfer_function(z, a_k, G_prime, N)
    assert isinstance(result, dict)
