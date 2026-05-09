"""Tests for rglyp.rangayyan_lyapunov."""
import numpy as np
import pytest
from moirais.fn.rglyp import rangayyan_lyapunov


def test_rglyp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_lyapunov(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rglyp_edge():
    """Test edge cases."""
    result = rangayyan_lyapunov(np.array([42.0]))
    assert result['n'] == 1
