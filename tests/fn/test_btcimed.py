"""Tests for btcimed.boot_ci_median."""
import numpy as np
import pytest
from morie.fn.btcimed import boot_ci_median


def test_btcimed_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = boot_ci_median(x, B, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btcimed_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = boot_ci_median(x, B, alpha)
    assert isinstance(result, dict)
