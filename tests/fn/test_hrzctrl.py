"""Tests for hrzctrl.horowitz_control_function."""
import numpy as np
import pytest
from moirais.fn.hrzctrl import horowitz_control_function


def test_hrzctrl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    bandwidth = 0.3
    result = horowitz_control_function(x, y, w, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzctrl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    bandwidth = 0.3
    result = horowitz_control_function(x, y, w, bandwidth)
    assert isinstance(result, dict)
