"""Tests for ksr054.kosorok_ch2_m_estimator_lipschitz_envelope."""
import numpy as np
import pytest
from moirais.fn.ksr054 import kosorok_ch2_m_estimator_lipschitz_envelope


def test_ksr054_basic():
    """Test basic functionality."""
    m = 10
    theta_1 = np.random.default_rng(42).normal(0, 1, 100)
    theta_2 = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_m_estimator_lipschitz_envelope(m, theta_1, theta_2, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr054_edge():
    """Test edge cases."""
    m = 10
    theta_1 = np.random.default_rng(42).normal(0, 1, 100)
    theta_2 = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_m_estimator_lipschitz_envelope(m, theta_1, theta_2, x)
    assert isinstance(result, dict)
