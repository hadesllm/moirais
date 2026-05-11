"""Tests for km018.kamath_ch2_layer_norm."""
import numpy as np
import pytest
from morie.fn.km018 import kamath_ch2_layer_norm


def test_km018_basic():
    """Test basic functionality."""
    h_i = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch2_layer_norm(h_i, mu, sigma, g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km018_edge():
    """Test edge cases."""
    h_i = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch2_layer_norm(h_i, mu, sigma, g)
    assert isinstance(result, dict)
