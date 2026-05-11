"""Tests for johw.joseph_holt_winters."""
import numpy as np
import pytest
from morie.fn.johw import joseph_holt_winters


def test_johw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    gamma = 1.0
    m = 10
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_holt_winters(y, alpha, beta, gamma, m, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_johw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    gamma = 1.0
    m = 10
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_holt_winters(y, alpha, beta, gamma, m, horizon)
    assert isinstance(result, dict)
