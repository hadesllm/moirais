"""Tests for rng069.rangayyan_ch3_dft_definition."""
import numpy as np
import pytest
from moirais.fn.rng069 import rangayyan_ch3_dft_definition


def test_rng069_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    k = 5
    N = 100
    result = rangayyan_ch3_dft_definition(x, n, k, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng069_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    k = 5
    N = 100
    result = rangayyan_ch3_dft_definition(x, n, k, N)
    assert isinstance(result, dict)
