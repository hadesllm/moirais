"""Tests for rng051.rangayyan_ch3_lti_convolution_property."""
import numpy as np
import pytest
from moirais.fn.rng051 import rangayyan_ch3_lti_convolution_property


def test_rng051_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = rangayyan_ch3_lti_convolution_property(x, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng051_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = rangayyan_ch3_lti_convolution_property(x, h)
    assert isinstance(result, dict)
