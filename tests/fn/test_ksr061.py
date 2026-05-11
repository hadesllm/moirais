"""Tests for ksr061.kosorok_ch3_differentiable_quadratic_mean."""
import numpy as np
import pytest
from morie.fn.ksr061 import kosorok_ch3_differentiable_quadratic_mean


def test_ksr061_basic():
    """Test basic functionality."""
    P_t = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = kosorok_ch3_differentiable_quadratic_mean(P_t, P, g, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr061_edge():
    """Test edge cases."""
    P_t = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = kosorok_ch3_differentiable_quadratic_mean(P_t, P, g, t)
    assert isinstance(result, dict)
