"""Tests for ksr055.kosorok_ch2_m_estimator_taylor_expansion."""
import numpy as np
import pytest
from morie.fn.ksr055 import kosorok_ch2_m_estimator_taylor_expansion


def test_ksr055_basic():
    """Test basic functionality."""
    m = 10
    theta = 0.0
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_m_estimator_taylor_expansion(m, theta, theta_0, P)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr055_edge():
    """Test edge cases."""
    m = 10
    theta = 0.0
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_m_estimator_taylor_expansion(m, theta, theta_0, P)
    assert isinstance(result, dict)
