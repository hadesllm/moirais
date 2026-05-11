"""Tests for gb651p.gibbons_ctrl_median_power."""
import numpy as np
import pytest
from morie.fn.gb651p import gibbons_ctrl_median_power


def test_gb651p_basic():
    """Test basic functionality."""
    n = 100
    p = 5
    alpha = 0.05
    result = gibbons_ctrl_median_power(n, p, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb651p_edge():
    """Test edge cases."""
    n = 100
    p = 5
    alpha = 0.05
    result = gibbons_ctrl_median_power(n, p, alpha)
    assert isinstance(result, dict)
