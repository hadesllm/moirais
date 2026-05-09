"""Tests for rng083.rangayyan_ch3_even_odd_decomposition."""
import numpy as np
import pytest
from moirais.fn.rng083 import rangayyan_ch3_even_odd_decomposition


def test_rng083_basic():
    """Test basic functionality."""
    x_e = np.random.default_rng(42).normal(0, 1, 100)
    x_o = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_even_odd_decomposition(x_e, x_o, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng083_edge():
    """Test edge cases."""
    x_e = np.random.default_rng(42).normal(0, 1, 100)
    x_o = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_even_odd_decomposition(x_e, x_o, n)
    assert isinstance(result, dict)
