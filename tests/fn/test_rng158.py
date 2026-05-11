"""Tests for rng158.rangayyan_ch3_lms_steepest_descent."""
import numpy as np
import pytest
from morie.fn.rng158 import rangayyan_ch3_lms_steepest_descent


def test_rng158_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    mu = 0.0
    n = 100
    result = rangayyan_ch3_lms_steepest_descent(w, mu, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng158_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    mu = 0.0
    n = 100
    result = rangayyan_ch3_lms_steepest_descent(w, mu, n)
    assert isinstance(result, dict)
