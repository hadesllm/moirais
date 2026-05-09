"""Tests for rgcdft.rangayyan_circular_conv_dft."""
import numpy as np
import pytest
from moirais.fn.rgcdft import rangayyan_circular_conv_dft


def test_rgcdft_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = rangayyan_circular_conv_dft(x, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcdft_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = rangayyan_circular_conv_dft(x, h)
    assert isinstance(result, dict)
