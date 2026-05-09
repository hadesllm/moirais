"""Tests for rgmavg.rangayyan_moving_average."""
import numpy as np
import pytest
from moirais.fn.rgmavg import rangayyan_moving_average


def test_rgmavg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_moving_average(x, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmavg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_moving_average(x, M)
    assert isinstance(result, dict)
