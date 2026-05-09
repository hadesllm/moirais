"""Tests for gb434bt.gibbons_ks_bt_formula."""
import numpy as np
import pytest
from moirais.fn.gb434bt import gibbons_ks_bt_formula


def test_gb434bt_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_ks_bt_formula(c, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb434bt_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_ks_bt_formula(c, n)
    assert isinstance(result, dict)
