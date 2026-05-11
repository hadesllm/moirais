"""Tests for ksr045.kosorok_ch2_functional_delta_bootstrap."""
import numpy as np
import pytest
from morie.fn.ksr045 import kosorok_ch2_functional_delta_bootstrap


def test_ksr045_basic():
    """Test basic functionality."""
    phi = np.random.default_rng(42).normal(0, 1, 100)
    X_n = np.random.default_rng(42).normal(0, 1, 100)
    X_hat_n = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    r_n = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_functional_delta_bootstrap(phi, X_n, X_hat_n, mu, r_n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr045_edge():
    """Test edge cases."""
    phi = np.random.default_rng(42).normal(0, 1, 100)
    X_n = np.random.default_rng(42).normal(0, 1, 100)
    X_hat_n = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    r_n = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_functional_delta_bootstrap(phi, X_n, X_hat_n, mu, r_n)
    assert isinstance(result, dict)
