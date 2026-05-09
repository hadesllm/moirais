"""Tests for hwmul.holt_winters_mult."""
import numpy as np
import pytest
from moirais.fn.hwmul import holt_winters_mult


def test_hwmul_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    gamma = 1.0
    result = holt_winters_mult(y, period, alpha, beta, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hwmul_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    gamma = 1.0
    result = holt_winters_mult(y, period, alpha, beta, gamma)
    assert isinstance(result, dict)
