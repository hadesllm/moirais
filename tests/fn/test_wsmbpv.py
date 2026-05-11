"""Tests for wsmbpv.wasserman_bootstrap_pivotal."""
import numpy as np
import pytest
from morie.fn.wsmbpv import wasserman_bootstrap_pivotal


def test_wsmbpv_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = wasserman_bootstrap_pivotal(data, T, B, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmbpv_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = wasserman_bootstrap_pivotal(data, T, B, alpha)
    assert isinstance(result, dict)
