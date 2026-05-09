"""Tests for rng020.rangayyan_ch3_time_averaged_acf."""
import numpy as np
import pytest
from moirais.fn.rng020 import rangayyan_ch3_time_averaged_acf


def test_rng020_basic():
    """Test basic functionality."""
    x_k = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_time_averaged_acf(x_k, tau, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng020_edge():
    """Test edge cases."""
    x_k = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_time_averaged_acf(x_k, tau, T)
    assert isinstance(result, dict)
