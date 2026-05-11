"""Tests for gb661c.gibbons_mw_ci."""
import numpy as np
import pytest
from morie.fn.gb661c import gibbons_mw_ci


def test_gb661c_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_mw_ci(x, y, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb661c_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_mw_ci(x, y, alpha)
    assert isinstance(result, dict)
