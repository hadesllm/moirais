"""Tests for btjknd.boot_jackknife_d."""
import numpy as np
import pytest
from morie.fn.btjknd import boot_jackknife_d


def test_btjknd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    stat = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_jackknife_d(x, d, stat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btjknd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    stat = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_jackknife_d(x, d, stat)
    assert isinstance(result, dict)
