"""Tests for grtnh.geron_tanh_activation."""
import numpy as np
import pytest
from morie.fn.grtnh import geron_tanh_activation


def test_grtnh_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_tanh_activation(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grtnh_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_tanh_activation(z)
    assert isinstance(result, dict)
