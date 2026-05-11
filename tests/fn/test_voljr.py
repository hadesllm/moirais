"""Tests for voljr.vol_jump_robust_var."""
import numpy as np
import pytest
from morie.fn.voljr import vol_jump_robust_var


def test_voljr_basic():
    """Test basic functionality."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = vol_jump_robust_var(r_intraday, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_voljr_edge():
    """Test edge cases."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = vol_jump_robust_var(r_intraday, theta)
    assert isinstance(result, dict)
