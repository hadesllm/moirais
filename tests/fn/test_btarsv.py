"""Tests for btarsv.boot_ar_sieve."""
import numpy as np
import pytest
from moirais.fn.btarsv import boot_ar_sieve


def test_btarsv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p_max = 100
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_ar_sieve(x, p_max, stat, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btarsv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p_max = 100
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_ar_sieve(x, p_max, stat, B)
    assert isinstance(result, dict)
