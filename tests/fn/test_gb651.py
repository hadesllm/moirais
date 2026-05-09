"""Tests for gb651.gibbons_ctrl_median."""
import numpy as np
import pytest
from moirais.fn.gb651 import gibbons_ctrl_median


def test_gb651_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_ctrl_median(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb651_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_ctrl_median(x, y)
    assert isinstance(result, dict)
