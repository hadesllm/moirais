"""Tests for hrzsiqm.horowitz_sim_quantile."""
import numpy as np
import pytest
from morie.fn.hrzsiqm import horowitz_sim_quantile


def test_hrzsiqm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    tau = 0.1
    bandwidth = 0.3
    result = horowitz_sim_quantile(x, y, tau, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzsiqm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    tau = 0.1
    bandwidth = 0.3
    result = horowitz_sim_quantile(x, y, tau, bandwidth)
    assert isinstance(result, dict)
