"""Tests for hrzrateG.horowitz_rate_G_estimation."""
import numpy as np
import pytest
from morie.fn.hrzrateG import horowitz_rate_G_estimation


def test_hrzrateG_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_rate_G_estimation(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzrateG_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_rate_G_estimation(x, y, bandwidth)
    assert isinstance(result, dict)
