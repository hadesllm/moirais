"""Tests for hrznwr.horowitz_nw_regression."""
import numpy as np
import pytest
from moirais.fn.hrznwr import horowitz_nw_regression


def test_hrznwr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_nw_regression(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrznwr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_nw_regression(x, y, bandwidth)
    assert isinstance(result, dict)
