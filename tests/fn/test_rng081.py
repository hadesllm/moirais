"""Tests for rng081.rangayyan_ch3_even_part."""
import numpy as np
import pytest
from moirais.fn.rng081 import rangayyan_ch3_even_part


def test_rng081_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_even_part(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng081_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_even_part(x, n)
    assert isinstance(result, dict)
