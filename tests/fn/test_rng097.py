"""Tests for rng097.rangayyan_ch3_ma_8point."""
import numpy as np
import pytest
from moirais.fn.rng097 import rangayyan_ch3_ma_8point


def test_rng097_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_ma_8point(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng097_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_ma_8point(x, n)
    assert isinstance(result, dict)
