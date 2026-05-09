"""Tests for rgenv.rangayyan_envelope."""
import numpy as np
import pytest
from moirais.fn.rgenv import rangayyan_envelope


def test_rgenv_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_envelope(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgenv_edge():
    """Test edge cases."""
    result = rangayyan_envelope(np.array([42.0]))
    assert result['n'] == 1
