"""Tests for btjkn.boot_jackknife."""
import numpy as np
import pytest
from morie.fn.btjkn import boot_jackknife


def test_btjkn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_jackknife(x, stat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btjkn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_jackknife(x, stat)
    assert isinstance(result, dict)
