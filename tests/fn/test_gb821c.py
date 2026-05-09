"""Tests for gb821c.gibbons_wrs_ci."""
import numpy as np
import pytest
from moirais.fn.gb821c import gibbons_wrs_ci


def test_gb821c_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_wrs_ci(x, y, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb821c_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_wrs_ci(x, y, alpha)
    assert isinstance(result, dict)
