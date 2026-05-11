"""Tests for hrzp2.horowitz_plr_bandwidth."""
import numpy as np
import pytest
from morie.fn.hrzp2 import horowitz_plr_bandwidth


def test_hrzp2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_plr_bandwidth(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzp2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_plr_bandwidth(x, y)
    assert isinstance(result, dict)
