"""Tests for rng117.rangayyan_ch3_three_point_central_diff_transfer_function."""
import numpy as np
import pytest
from morie.fn.rng117 import rangayyan_ch3_three_point_central_diff_transfer_function


def test_rng117_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_three_point_central_diff_transfer_function(z, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng117_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_three_point_central_diff_transfer_function(z, T)
    assert isinstance(result, dict)
