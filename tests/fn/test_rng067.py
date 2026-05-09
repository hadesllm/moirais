"""Tests for rng067.rangayyan_ch3_dtft."""
import numpy as np
import pytest
from moirais.fn.rng067 import rangayyan_ch3_dtft


def test_rng067_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_dtft(x, n, omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng067_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_dtft(x, n, omega)
    assert isinstance(result, dict)
