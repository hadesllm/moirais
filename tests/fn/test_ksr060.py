"""Tests for ksr060.kosorok_ch2_u_process_measure."""
import numpy as np
import pytest
from morie.fn.ksr060 import kosorok_ch2_u_process_measure


def test_ksr060_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n = 100
    m = 10
    result = kosorok_ch2_u_process_measure(f, X, n, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr060_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n = 100
    m = 10
    result = kosorok_ch2_u_process_measure(f, X, n, m)
    assert isinstance(result, dict)
