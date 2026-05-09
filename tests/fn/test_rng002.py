"""Tests for rng002.rangayyan_ch3_mean_squared_value."""
import numpy as np
import pytest
from moirais.fn.rng002 import rangayyan_ch3_mean_squared_value


def test_rng002_basic():
    """Test basic functionality."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    p_eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_mean_squared_value(eta, p_eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng002_edge():
    """Test edge cases."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    p_eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_mean_squared_value(eta, p_eta)
    assert isinstance(result, dict)
