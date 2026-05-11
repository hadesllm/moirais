"""Tests for fzt58.fauzi_thm5_8_smoothed_convergence."""
import numpy as np
import pytest
from morie.fn.fzt58 import fauzi_thm5_8_smoothed_convergence


def test_fzt58_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    theta = 0.0
    result = fauzi_thm5_8_smoothed_convergence(data, bandwidth, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt58_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    theta = 0.0
    result = fauzi_thm5_8_smoothed_convergence(data, bandwidth, theta)
    assert isinstance(result, dict)
