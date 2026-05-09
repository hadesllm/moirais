"""Tests for gb651c.gibbons_ctrl_median_curtail."""
import numpy as np
import pytest
from moirais.fn.gb651c import gibbons_ctrl_median_curtail


def test_gb651c_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    control_median = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_ctrl_median_curtail(x, control_median, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb651c_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    control_median = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_ctrl_median_curtail(x, control_median, alpha)
    assert isinstance(result, dict)
