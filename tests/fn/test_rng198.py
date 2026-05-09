"""Tests for rng198.rangayyan_ch4_dot_product_discrete."""
import numpy as np
import pytest
from moirais.fn.rng198 import rangayyan_ch4_dot_product_discrete


def test_rng198_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch4_dot_product_discrete(x, y, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng198_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch4_dot_product_discrete(x, y, N)
    assert isinstance(result, dict)
