"""Tests for rgqrs.rangayyan_qrs_detect."""
import numpy as np
import pytest
from moirais.fn.rgqrs import rangayyan_qrs_detect


def test_rgqrs_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_qrs_detect(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgqrs_edge():
    """Test edge cases."""
    result = rangayyan_qrs_detect(np.array([42.0]))
    assert result['n'] == 1
