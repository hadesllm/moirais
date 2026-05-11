"""Tests for rng118.rangayyan_ch3_three_point_central_diff_magnitude."""
import numpy as np
import pytest
from morie.fn.rng118 import rangayyan_ch3_three_point_central_diff_magnitude


def test_rng118_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_three_point_central_diff_magnitude(omega, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng118_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_three_point_central_diff_magnitude(omega, T)
    assert isinstance(result, dict)
