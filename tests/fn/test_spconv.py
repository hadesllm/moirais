"""Tests for spconv.schabenberger_convolution_representation."""
import numpy as np
import pytest
from morie.fn.spconv import schabenberger_convolution_representation


def test_spconv_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = schabenberger_convolution_representation(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_spconv_edge():
    """Test edge cases."""
    result = schabenberger_convolution_representation(np.array([42.0]))
    assert result['n'] == 1
