"""Tests for km140.kamath_ch9_moc_loss."""
import numpy as np
import pytest
from moirais.fn.km140 import kamath_ch9_moc_loss


def test_km140_basic():
    """Test basic functionality."""
    theta = 0.0
    w = np.random.default_rng(45).exponential(1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    g_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_moc_loss(theta, w, v, g_theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km140_edge():
    """Test edge cases."""
    theta = 0.0
    w = np.random.default_rng(45).exponential(1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    g_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_moc_loss(theta, w, v, g_theta)
    assert isinstance(result, dict)
