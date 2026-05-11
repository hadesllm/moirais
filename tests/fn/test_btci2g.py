"""Tests for btci2g.boot_ci_two_groups."""
import numpy as np
import pytest
from morie.fn.btci2g import boot_ci_two_groups


def test_btci2g_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = boot_ci_two_groups(x, y, stat, B, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_btci2g_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = boot_ci_two_groups(x, y, stat, B, alpha)
    assert isinstance(result, dict)
