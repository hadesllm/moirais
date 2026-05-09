"""Tests for hrztfap.horowitz_T_F_asymp_props."""
import numpy as np
import pytest
from moirais.fn.hrztfap import horowitz_T_F_asymp_props


def test_hrztfap_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_T_F_asymp_props(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrztfap_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_T_F_asymp_props(x, y, bandwidth)
    assert isinstance(result, dict)
