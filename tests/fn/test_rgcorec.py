"""Tests for rgcorec.rangayyan_correlation_coeff."""
import numpy as np
import pytest
from morie.fn.rgcorec import rangayyan_correlation_coeff


def test_rgcorec_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = rangayyan_correlation_coeff(x, y)
    assert abs(result['statistic'] - 1.0) < 0.01


def test_rgcorec_edge():
    """Test edge cases."""
    result = rangayyan_correlation_coeff(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result['n'] == 2
