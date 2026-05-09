"""Tests for gb1331s.gibbons_sign_efficacy."""
import numpy as np
import pytest
from moirais.fn.gb1331s import gibbons_sign_efficacy


def test_gb1331s_basic():
    """Test basic functionality."""
    N = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    median = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_sign_efficacy(N, f, median)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1331s_edge():
    """Test edge cases."""
    N = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    median = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_sign_efficacy(N, f, median)
    assert isinstance(result, dict)
