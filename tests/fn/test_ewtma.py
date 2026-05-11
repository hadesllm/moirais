"""Tests for ewtma.ewma_volatility."""
import numpy as np
import pytest
from morie.fn.ewtma import ewma_volatility


def test_ewtma_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ewma_volatility(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ewtma_edge():
    """Test edge cases."""
    result = ewma_volatility(np.array([42.0]))
    assert result['n'] == 1
