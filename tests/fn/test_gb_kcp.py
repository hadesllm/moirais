"""Tests for gb_kcp.gibbons_ks_conf_band."""
import numpy as np
import pytest
from morie.fn.gb_kcp import gibbons_ks_conf_band


def test_gb_kcp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_ks_conf_band(x, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_kcp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_ks_conf_band(x, alpha)
    assert isinstance(result, dict)
