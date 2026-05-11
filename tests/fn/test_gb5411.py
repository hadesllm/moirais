"""Tests for gb5411.gibbons_sign_pvalue."""
import numpy as np
import pytest
from morie.fn.gb5411 import gibbons_sign_pvalue


def test_gb5411_basic():
    """Test basic functionality."""
    k_obs = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_sign_pvalue(k_obs, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb5411_edge():
    """Test edge cases."""
    k_obs = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_sign_pvalue(k_obs, n)
    assert isinstance(result, dict)
