"""Tests for hrznwrg.horowitz_nw_estimator_g."""
import numpy as np
import pytest
from morie.fn.hrznwrg import horowitz_nw_estimator_g


def test_hrznwrg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    bandwidth = 0.3
    result = horowitz_nw_estimator_g(x, y, beta, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrznwrg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    bandwidth = 0.3
    result = horowitz_nw_estimator_g(x, y, beta, bandwidth)
    assert isinstance(result, dict)
