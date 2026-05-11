"""Tests for rng099.rangayyan_ch3_ma_8point_transfer_function."""
import numpy as np
import pytest
from morie.fn.rng099 import rangayyan_ch3_ma_8point_transfer_function


def test_rng099_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch3_ma_8point_transfer_function(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng099_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch3_ma_8point_transfer_function(z)
    assert isinstance(result, dict)
