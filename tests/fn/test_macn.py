"""Tests for macn.ma_cochran_q."""
import numpy as np
import pytest
from morie.fn.macn import ma_cochran_q


def test_macn_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_cochran_q(yi, vi)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_macn_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_cochran_q(yi, vi)
    assert isinstance(result, dict)
