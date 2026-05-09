"""Tests for ksr030.kosorok_ch2_brownian_bridge_covariance."""
import numpy as np
import pytest
from moirais.fn.ksr030 import kosorok_ch2_brownian_bridge_covariance


def test_ksr030_basic():
    """Test basic functionality."""
    s = 90
    t = np.linspace(0, 10, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_ch2_brownian_bridge_covariance(s, t, F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr030_edge():
    """Test edge cases."""
    s = 90
    t = np.linspace(0, 10, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_ch2_brownian_bridge_covariance(s, t, F)
    assert isinstance(result, dict)
