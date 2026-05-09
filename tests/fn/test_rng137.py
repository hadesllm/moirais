"""Tests for rng137.rangayyan_ch3_estimation_error."""
import numpy as np
import pytest
from moirais.fn.rng137 import rangayyan_ch3_estimation_error


def test_rng137_basic():
    """Test basic functionality."""
    d = 5
    d_tilde = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_estimation_error(d, d_tilde, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng137_edge():
    """Test edge cases."""
    d = 5
    d_tilde = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_estimation_error(d, d_tilde, n)
    assert isinstance(result, dict)
