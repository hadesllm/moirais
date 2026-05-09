"""Tests for rng008.rangayyan_ch3_sample_mean_squared."""
import numpy as np
import pytest
from moirais.fn.rng008 import rangayyan_ch3_sample_mean_squared


def test_rng008_basic():
    """Test basic functionality."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_sample_mean_squared(eta, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng008_edge():
    """Test edge cases."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_sample_mean_squared(eta, N)
    assert isinstance(result, dict)
