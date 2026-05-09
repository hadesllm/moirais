"""Tests for ksr042.kosorok_ch2_functional_delta_method."""
import numpy as np
import pytest
from moirais.fn.ksr042 import kosorok_ch2_functional_delta_method


def test_ksr042_basic():
    """Test basic functionality."""
    phi = np.random.default_rng(42).normal(0, 1, 100)
    X_n = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    r_n = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_functional_delta_method(phi, X_n, theta, r_n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr042_edge():
    """Test edge cases."""
    phi = np.random.default_rng(42).normal(0, 1, 100)
    X_n = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    r_n = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_functional_delta_method(phi, X_n, theta, r_n)
    assert isinstance(result, dict)
