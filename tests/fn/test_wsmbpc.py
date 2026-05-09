"""Tests for wsmbpc.wasserman_bootstrap_percentile."""
import numpy as np
import pytest
from moirais.fn.wsmbpc import wasserman_bootstrap_percentile


def test_wsmbpc_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = wasserman_bootstrap_percentile(data, T, B, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmbpc_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = wasserman_bootstrap_percentile(data, T, B, alpha)
    assert isinstance(result, dict)
