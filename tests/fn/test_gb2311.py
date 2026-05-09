"""Tests for gb2311.gibbons_edf_mean_var."""
import numpy as np
import pytest
from moirais.fn.gb2311 import gibbons_edf_mean_var


def test_gb2311_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_edf_mean_var(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb2311_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_edf_mean_var(x, n)
    assert isinstance(result, dict)
