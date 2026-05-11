"""Tests for km141.kamath_ch9_itm_loss."""
import numpy as np
import pytest
from morie.fn.km141 import kamath_ch9_itm_loss


def test_km141_basic():
    """Test basic functionality."""
    theta = 0.0
    v = np.random.default_rng(44).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch9_itm_loss(theta, v, t, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km141_edge():
    """Test edge cases."""
    theta = 0.0
    v = np.random.default_rng(44).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch9_itm_loss(theta, v, t, y)
    assert isinstance(result, dict)
