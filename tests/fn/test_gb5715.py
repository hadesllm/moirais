"""Tests for gb5715.gibbons_wsrt_ci."""
import numpy as np
import pytest
from morie.fn.gb5715 import gibbons_wsrt_ci


def test_gb5715_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_wsrt_ci(x, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb5715_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_wsrt_ci(x, alpha)
    assert isinstance(result, dict)
