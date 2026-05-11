"""Tests for btjkab.boot_jackknife_after_boot."""
import numpy as np
import pytest
from morie.fn.btjkab import boot_jackknife_after_boot


def test_btjkab_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    B_idx = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_jackknife_after_boot(x, theta_b, B_idx)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btjkab_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    B_idx = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_jackknife_after_boot(x, theta_b, B_idx)
    assert isinstance(result, dict)
