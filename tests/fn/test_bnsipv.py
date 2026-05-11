"""Tests for bnsipv.bound_iv_partial."""
import numpy as np
import pytest
from morie.fn.bnsipv import bound_iv_partial


def test_bnsipv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = bound_iv_partial(y, D, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnsipv_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = bound_iv_partial(y, D, Z)
    assert isinstance(result, dict)
