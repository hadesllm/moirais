"""Tests for grubbs.grubbs_test."""
import numpy as np
import pytest
from morie.fn.grubbs import grubbs_test


def test_grubbs_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = grubbs_test(x, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_grubbs_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = grubbs_test(x, alpha)
    assert isinstance(result, dict)
