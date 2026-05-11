"""Tests for hmswin.geron_swin."""
import numpy as np
import pytest
from morie.fn.hmswin import geron_swin


def test_hmswin_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    window_size = 100
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_swin(image, window_size, n_layers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmswin_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    window_size = 100
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_swin(image, window_size, n_layers)
    assert isinstance(result, dict)
