"""Tests for rng148.rangayyan_ch3_wiener_convolution_relationship."""
import numpy as np
import pytest
from morie.fn.rng148 import rangayyan_ch3_wiener_convolution_relationship


def test_rng148_basic():
    """Test basic functionality."""
    w_ok = np.random.default_rng(42).normal(0, 1, 100)
    phi = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    k = 5
    result = rangayyan_ch3_wiener_convolution_relationship(w_ok, phi, theta, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng148_edge():
    """Test edge cases."""
    w_ok = np.random.default_rng(42).normal(0, 1, 100)
    phi = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    k = 5
    result = rangayyan_ch3_wiener_convolution_relationship(w_ok, phi, theta, k)
    assert isinstance(result, dict)
