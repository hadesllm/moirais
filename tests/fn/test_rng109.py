"""Tests for rng109.rangayyan_ch3_ma_8point_recursive_transfer_function."""
import numpy as np
import pytest
from moirais.fn.rng109 import rangayyan_ch3_ma_8point_recursive_transfer_function


def test_rng109_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch3_ma_8point_recursive_transfer_function(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng109_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch3_ma_8point_recursive_transfer_function(z)
    assert isinstance(result, dict)
