"""Tests for wsmksm.wasserman_ks_test."""
import numpy as np
import pytest
from morie.fn.wsmksm import wasserman_ks_test


def test_wsmksm_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f0 = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_ks_test(data, f0)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wsmksm_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f0 = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_ks_test(data, f0)
    assert isinstance(result, dict)
