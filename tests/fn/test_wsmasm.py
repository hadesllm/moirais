"""Tests for wsmasm.wasserman_mle_asymptotic."""
import numpy as np
import pytest
from morie.fn.wsmasm import wasserman_mle_asymptotic


def test_wsmasm_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_mle_asymptotic(data, f, theta_hat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmasm_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_mle_asymptotic(data, f, theta_hat)
    assert isinstance(result, dict)
