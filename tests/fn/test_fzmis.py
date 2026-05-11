"""Tests for fzmis.fauzi_mise_computation."""
import numpy as np
import pytest
from morie.fn.fzmis import fauzi_mise_computation


def test_fzmis_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_mise_computation(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_fzmis_edge():
    """Test edge cases."""
    result = fauzi_mise_computation(np.array([42.0]))
    assert result['n'] == 1
