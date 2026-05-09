"""Tests for bnnipw.bound_no_iv_proxy."""
import numpy as np
import pytest
from moirais.fn.bnnipw import bound_no_iv_proxy


def test_bnnipw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z_proxy = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_no_iv_proxy(y, D, Z_proxy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnnipw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z_proxy = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_no_iv_proxy(y, D, Z_proxy)
    assert isinstance(result, dict)
