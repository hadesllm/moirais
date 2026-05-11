"""Tests for btdbl.boot_double."""
import numpy as np
import pytest
from morie.fn.btdbl import boot_double


def test_btdbl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    Bp = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_double(x, stat, B, Bp, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btdbl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    Bp = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_double(x, stat, B, Bp, alpha)
    assert isinstance(result, dict)
