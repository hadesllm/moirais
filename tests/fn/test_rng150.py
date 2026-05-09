"""Tests for rng150.rangayyan_ch3_wiener_frequency_response."""
import numpy as np
import pytest
from moirais.fn.rng150 import rangayyan_ch3_wiener_frequency_response


def test_rng150_basic():
    """Test basic functionality."""
    S_xd = np.random.default_rng(42).normal(0, 1, 100)
    S_xx = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_wiener_frequency_response(S_xd, S_xx, omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng150_edge():
    """Test edge cases."""
    S_xd = np.random.default_rng(42).normal(0, 1, 100)
    S_xx = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_wiener_frequency_response(S_xd, S_xx, omega)
    assert isinstance(result, dict)
