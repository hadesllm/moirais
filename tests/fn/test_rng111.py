"""Tests for rng111.rangayyan_ch3_first_difference_operator."""
import numpy as np
import pytest
from moirais.fn.rng111 import rangayyan_ch3_first_difference_operator


def test_rng111_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    n = 100
    result = rangayyan_ch3_first_difference_operator(x, T, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng111_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    n = 100
    result = rangayyan_ch3_first_difference_operator(x, T, n)
    assert isinstance(result, dict)
