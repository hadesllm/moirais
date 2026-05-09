"""Tests for ksr059.kosorok_ch2_kmt_strong_approximation."""
import numpy as np
import pytest
from moirais.fn.ksr059 import kosorok_ch2_kmt_strong_approximation


def test_ksr059_basic():
    """Test basic functionality."""
    G_n = np.random.default_rng(42).normal(0, 1, 100)
    B_n = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_kmt_strong_approximation(G_n, B_n, F, n, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr059_edge():
    """Test edge cases."""
    G_n = np.random.default_rng(42).normal(0, 1, 100)
    B_n = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_kmt_strong_approximation(G_n, B_n, F, n, x)
    assert isinstance(result, dict)
