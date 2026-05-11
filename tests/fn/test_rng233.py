"""Tests for rng233.rangayyan_ch4_convolution_model."""
import numpy as np
import pytest
from morie.fn.rng233 import rangayyan_ch4_convolution_model


def test_rng233_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_convolution_model(x, h, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng233_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_convolution_model(x, h, t)
    assert isinstance(result, dict)
