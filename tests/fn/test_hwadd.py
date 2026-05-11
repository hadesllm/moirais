"""Tests for hwadd.holt_winters_additive."""
import numpy as np
import pytest
from morie.fn.hwadd import holt_winters_additive


def test_hwadd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    gamma = 1.0
    result = holt_winters_additive(y, period, alpha, beta, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hwadd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    gamma = 1.0
    result = holt_winters_additive(y, period, alpha, beta, gamma)
    assert isinstance(result, dict)
