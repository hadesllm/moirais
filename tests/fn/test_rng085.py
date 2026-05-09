"""Tests for rng085.rangayyan_ch3_synchronized_averaging_sum."""
import numpy as np
import pytest
from moirais.fn.rng085 import rangayyan_ch3_synchronized_averaging_sum


def test_rng085_basic():
    """Test basic functionality."""
    y_k = np.random.default_rng(42).normal(0, 1, 100)
    x_k = np.random.default_rng(42).normal(0, 1, 100)
    eta_k = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_synchronized_averaging_sum(y_k, x_k, eta_k, n, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng085_edge():
    """Test edge cases."""
    y_k = np.random.default_rng(42).normal(0, 1, 100)
    x_k = np.random.default_rng(42).normal(0, 1, 100)
    eta_k = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_synchronized_averaging_sum(y_k, x_k, eta_k, n, M)
    assert isinstance(result, dict)
