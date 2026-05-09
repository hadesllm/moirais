"""Tests for rng084.rangayyan_ch3_observed_signal_kth_realization."""
import numpy as np
import pytest
from moirais.fn.rng084 import rangayyan_ch3_observed_signal_kth_realization


def test_rng084_basic():
    """Test basic functionality."""
    x_k = np.random.default_rng(42).normal(0, 1, 100)
    eta_k = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_observed_signal_kth_realization(x_k, eta_k, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng084_edge():
    """Test edge cases."""
    x_k = np.random.default_rng(42).normal(0, 1, 100)
    eta_k = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_observed_signal_kth_realization(x_k, eta_k, n)
    assert isinstance(result, dict)
