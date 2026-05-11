"""Tests for vtpwr.voting_power_index."""
import numpy as np
import pytest
from morie.fn.vtpwr import voting_power_index


def test_vtpwr_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = voting_power_index(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_vtpwr_edge():
    """Test edge cases."""
    result = voting_power_index(np.array([42.0]))
    assert result['n'] == 1
