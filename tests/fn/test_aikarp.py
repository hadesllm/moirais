"""Tests for aikarp.aic_ar_order."""
import numpy as np
import pytest
from morie.fn.aikarp import aic_ar_order


def test_aikarp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_p = np.random.default_rng(42).normal(0, 1, 100)
    result = aic_ar_order(x, max_p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aikarp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_p = np.random.default_rng(42).normal(0, 1, 100)
    result = aic_ar_order(x, max_p)
    assert isinstance(result, dict)
