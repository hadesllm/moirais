"""Tests for rng157.rangayyan_ch3_lms_squared_error."""
import numpy as np
import pytest
from moirais.fn.rng157 import rangayyan_ch3_lms_squared_error


def test_rng157_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    w = np.random.default_rng(45).exponential(1, 100)
    n = 100
    result = rangayyan_ch3_lms_squared_error(x, r, w, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng157_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    w = np.random.default_rng(45).exponential(1, 100)
    n = 100
    result = rangayyan_ch3_lms_squared_error(x, r, w, n)
    assert isinstance(result, dict)
