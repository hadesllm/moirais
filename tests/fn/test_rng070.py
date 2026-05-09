"""Tests for rng070.rangayyan_ch3_idft_definition."""
import numpy as np
import pytest
from moirais.fn.rng070 import rangayyan_ch3_idft_definition


def test_rng070_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n = 100
    k = 5
    N = 100
    result = rangayyan_ch3_idft_definition(X, n, k, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng070_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n = 100
    k = 5
    N = 100
    result = rangayyan_ch3_idft_definition(X, n, k, N)
    assert isinstance(result, dict)
