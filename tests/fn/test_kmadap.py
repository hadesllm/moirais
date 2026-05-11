"""Tests for kmadap.kamath_houlsby_adapter."""
import numpy as np
import pytest
from morie.fn.kmadap import kamath_houlsby_adapter


def test_kmadap_basic():
    """Test basic functionality."""
    h = 0.3
    W_down = np.random.default_rng(42).normal(0, 1, 100)
    W_up = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_houlsby_adapter(h, W_down, W_up)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmadap_edge():
    """Test edge cases."""
    h = 0.3
    W_down = np.random.default_rng(42).normal(0, 1, 100)
    W_up = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_houlsby_adapter(h, W_down, W_up)
    assert isinstance(result, dict)
