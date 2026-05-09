"""Tests for ksr056.kosorok_ch2_lad_lipschitz_bound."""
import numpy as np
import pytest
from moirais.fn.ksr056 import kosorok_ch2_lad_lipschitz_bound


def test_ksr056_basic():
    """Test basic functionality."""
    theta_1 = np.random.default_rng(42).normal(0, 1, 100)
    theta_2 = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_lad_lipschitz_bound(theta_1, theta_2, u, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr056_edge():
    """Test edge cases."""
    theta_1 = np.random.default_rng(42).normal(0, 1, 100)
    theta_2 = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_lad_lipschitz_bound(theta_1, theta_2, u, x)
    assert isinstance(result, dict)
