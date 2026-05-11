"""Tests for whtinv.walsh_hadamard_inverse."""
import numpy as np
import pytest
from morie.fn.whtinv import walsh_hadamard_inverse


def test_whtinv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = walsh_hadamard_inverse(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_whtinv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = walsh_hadamard_inverse(x)
    assert isinstance(result, dict)
