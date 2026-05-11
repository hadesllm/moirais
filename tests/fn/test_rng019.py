"""Tests for rng019.rangayyan_ch3_time_average_mean."""
import numpy as np
import pytest
from morie.fn.rng019 import rangayyan_ch3_time_average_mean


def test_rng019_basic():
    """Test basic functionality."""
    x_k = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_time_average_mean(x_k, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng019_edge():
    """Test edge cases."""
    x_k = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_time_average_mean(x_k, T)
    assert isinstance(result, dict)
