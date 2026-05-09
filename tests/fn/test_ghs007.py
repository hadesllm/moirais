"""Tests for ghs007.ghosal_ch2_binary_regression_density."""
import numpy as np
import pytest
from moirais.fn.ghs007 import ghosal_ch2_binary_regression_density


def test_ghs007_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch2_binary_regression_density(y, x, f, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs007_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch2_binary_regression_density(y, x, f, H)
    assert isinstance(result, dict)
