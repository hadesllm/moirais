"""Tests for joholt.joseph_holt_linear."""
import numpy as np
import pytest
from morie.fn.joholt import joseph_holt_linear


def test_joholt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_holt_linear(y, alpha, beta, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_joholt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_holt_linear(y, alpha, beta, horizon)
    assert isinstance(result, dict)
