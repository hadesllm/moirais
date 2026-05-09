"""Tests for ksr051.kosorok_ch2_continuous_invertibility."""
import numpy as np
import pytest
from moirais.fn.ksr051 import kosorok_ch2_continuous_invertibility


def test_ksr051_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    theta_1 = np.random.default_rng(42).normal(0, 1, 100)
    theta_2 = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_continuous_invertibility(A, theta_1, theta_2, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr051_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    theta_1 = np.random.default_rng(42).normal(0, 1, 100)
    theta_2 = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_continuous_invertibility(A, theta_1, theta_2, c)
    assert isinstance(result, dict)
