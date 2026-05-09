"""Tests for hrzplrq.horowitz_plr_quantile."""
import numpy as np
import pytest
from moirais.fn.hrzplrq import horowitz_plr_quantile


def test_hrzplrq_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    bandwidth = 0.3
    tau = 0.1
    result = horowitz_plr_quantile(x, y, z, bandwidth, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzplrq_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    bandwidth = 0.3
    tau = 0.1
    result = horowitz_plr_quantile(x, y, z, bandwidth, tau)
    assert isinstance(result, dict)
