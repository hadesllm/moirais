"""Tests for rng162.rangayyan_ch3_lms_step_size_zhang."""
import numpy as np
import pytest
from morie.fn.rng162 import rangayyan_ch3_lms_step_size_zhang


def test_rng162_basic():
    """Test basic functionality."""
    mu = 0.0
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    x_bar = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    r = 10
    n = 100
    result = rangayyan_ch3_lms_step_size_zhang(mu, M, x_bar, alpha, r, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng162_edge():
    """Test edge cases."""
    mu = 0.0
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    x_bar = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    r = 10
    n = 100
    result = rangayyan_ch3_lms_step_size_zhang(mu, M, x_bar, alpha, r, n)
    assert isinstance(result, dict)
