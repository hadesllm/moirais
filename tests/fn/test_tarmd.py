"""Tests for tarmd.threshold_autoregression."""
import numpy as np
import pytest
from moirais.fn.tarmd import threshold_autoregression


def test_tarmd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = threshold_autoregression(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_tarmd_edge():
    """Test edge cases."""
    result = threshold_autoregression(np.array([42.0]))
    assert result['n'] == 1
