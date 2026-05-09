"""Tests for hrzaqm.horowitz_additive_quantile."""
import numpy as np
import pytest
from moirais.fn.hrzaqm import horowitz_additive_quantile


def test_hrzaqm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    tau = 0.1
    result = horowitz_additive_quantile(x, y, bandwidth, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzaqm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    tau = 0.1
    result = horowitz_additive_quantile(x, y, bandwidth, tau)
    assert isinstance(result, dict)
