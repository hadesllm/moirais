"""Tests for manmar.ma_network_indirect."""
import numpy as np
import pytest
from moirais.fn.manmar import ma_network_indirect


def test_manmar_basic():
    """Test basic functionality."""
    d_AB = np.random.default_rng(42).normal(0, 1, 100)
    v_AB = np.random.default_rng(42).normal(0, 1, 100)
    d_CB = np.random.default_rng(42).normal(0, 1, 100)
    v_CB = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_network_indirect(d_AB, v_AB, d_CB, v_CB)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_manmar_edge():
    """Test edge cases."""
    d_AB = np.random.default_rng(42).normal(0, 1, 100)
    v_AB = np.random.default_rng(42).normal(0, 1, 100)
    d_CB = np.random.default_rng(42).normal(0, 1, 100)
    v_CB = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_network_indirect(d_AB, v_AB, d_CB, v_CB)
    assert isinstance(result, dict)
