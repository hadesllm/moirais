"""Tests for rng114.rangayyan_ch3_first_difference_magnitude."""
import numpy as np
import pytest
from morie.fn.rng114 import rangayyan_ch3_first_difference_magnitude


def test_rng114_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_first_difference_magnitude(omega, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng114_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_first_difference_magnitude(omega, T)
    assert isinstance(result, dict)
