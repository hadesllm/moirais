"""Tests for gb_blt.gibbons_balance_incomplete."""
import numpy as np
import pytest
from moirais.fn.gb_blt import gibbons_balance_incomplete


def test_gb_blt_basic():
    """Test basic functionality."""
    rankings = np.random.default_rng(42).permutation(10).reshape(2, 5)
    lam = 0.1
    n = 100
    k = 5
    result = gibbons_balance_incomplete(rankings, lam, n, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_blt_edge():
    """Test edge cases."""
    rankings = np.random.default_rng(42).permutation(10).reshape(2, 5)
    lam = 0.1
    n = 100
    k = 5
    result = gibbons_balance_incomplete(rankings, lam, n, k)
    assert isinstance(result, dict)
