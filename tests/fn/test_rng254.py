"""Tests for rng254.rangayyan_ch4_power_cepstrum_sum."""
import numpy as np
import pytest
from moirais.fn.rng254 import rangayyan_ch4_power_cepstrum_sum


def test_rng254_basic():
    """Test basic functionality."""
    x_hat_p = np.random.default_rng(42).normal(0, 1, 100)
    h_hat_p = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_power_cepstrum_sum(x_hat_p, h_hat_p, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng254_edge():
    """Test edge cases."""
    x_hat_p = np.random.default_rng(42).normal(0, 1, 100)
    h_hat_p = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_power_cepstrum_sum(x_hat_p, h_hat_p, n)
    assert isinstance(result, dict)
