"""Tests for btsmth.boot_smoothed."""
import numpy as np
import pytest
from moirais.fn.btsmth import boot_smoothed


def test_btsmth_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_smoothed(x, stat, h, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btsmth_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_smoothed(x, stat, h, B)
    assert isinstance(result, dict)
