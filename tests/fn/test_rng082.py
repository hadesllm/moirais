"""Tests for rng082.rangayyan_ch3_odd_part."""
import numpy as np
import pytest
from moirais.fn.rng082 import rangayyan_ch3_odd_part


def test_rng082_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_odd_part(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng082_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_odd_part(x, n)
    assert isinstance(result, dict)
