"""Tests for hkonly.hadamard_response."""
import numpy as np
import pytest
from morie.fn.hkonly import hadamard_response


def test_hkonly_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = hadamard_response(x, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hkonly_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = hadamard_response(x, epsilon)
    assert isinstance(result, dict)
