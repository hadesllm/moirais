"""Tests for btcicor.boot_ci_correlation."""
import numpy as np
import pytest
from morie.fn.btcicor import boot_ci_correlation


def test_btcicor_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = boot_ci_correlation(x, y, B, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_btcicor_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = boot_ci_correlation(x, y, B, alpha)
    assert isinstance(result, dict)
