"""Tests for rng113.rangayyan_ch3_first_difference_frequency_response."""
import numpy as np
import pytest
from morie.fn.rng113 import rangayyan_ch3_first_difference_frequency_response


def test_rng113_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_first_difference_frequency_response(omega, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng113_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_first_difference_frequency_response(omega, T)
    assert isinstance(result, dict)
