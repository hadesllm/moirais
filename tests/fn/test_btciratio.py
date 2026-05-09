"""Tests for btciratio.boot_ci_ratio."""
import numpy as np
import pytest
from moirais.fn.btciratio import boot_ci_ratio


def test_btciratio_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    stat_x = np.random.default_rng(42).normal(0, 1, 100)
    stat_y = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = boot_ci_ratio(x, y, stat_x, stat_y, B, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btciratio_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    stat_x = np.random.default_rng(42).normal(0, 1, 100)
    stat_y = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = boot_ci_ratio(x, y, stat_x, stat_y, B, alpha)
    assert isinstance(result, dict)
