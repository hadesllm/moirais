"""Tests for nbeat.nbeats_basis."""
import numpy as np
import pytest
from morie.fn.nbeat import nbeats_basis


def test_nbeat_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = nbeats_basis(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_nbeat_edge():
    """Test edge cases."""
    result = nbeats_basis(np.array([42.0]))
    assert result['n'] == 1
