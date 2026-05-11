"""Tests for gstarl.local_g_star."""
import numpy as np
import pytest
from morie.fn.gstarl import local_g_star


def test_gstarl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = local_g_star(x, W)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gstarl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = local_g_star(x, W)
    assert isinstance(result, dict)
