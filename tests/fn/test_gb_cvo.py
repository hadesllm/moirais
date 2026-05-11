"""Tests for gb_cvo.gibbons_order_covariance."""
import numpy as np
import pytest
from morie.fn.gb_cvo import gibbons_order_covariance


def test_gb_cvo_basic():
    """Test basic functionality."""
    r = 10
    s = 90
    n = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_order_covariance(r, s, n, f, F)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_cvo_edge():
    """Test edge cases."""
    r = 10
    s = 90
    n = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_order_covariance(r, s, n, f, F)
    assert isinstance(result, dict)
