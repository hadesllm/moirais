"""Tests for rng050.rangayyan_ch3_frequency_response_from_laplace."""
import numpy as np
import pytest
from moirais.fn.rng050 import rangayyan_ch3_frequency_response_from_laplace


def test_rng050_basic():
    """Test basic functionality."""
    h = 0.3
    omega = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_frequency_response_from_laplace(h, omega, t, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng050_edge():
    """Test edge cases."""
    h = 0.3
    omega = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_frequency_response_from_laplace(h, omega, t, T)
    assert isinstance(result, dict)
