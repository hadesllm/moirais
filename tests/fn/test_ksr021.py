"""Tests for ksr021.kosorok_ch1_residual_empirical_distribution."""
import numpy as np
import pytest
from moirais.fn.ksr021 import kosorok_ch1_residual_empirical_distribution


def test_ksr021_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    beta_hat = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    n = 100
    result = kosorok_ch1_residual_empirical_distribution(Y, Z, beta_hat, t, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr021_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    beta_hat = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    n = 100
    result = kosorok_ch1_residual_empirical_distribution(Y, Z, beta_hat, t, n)
    assert isinstance(result, dict)
