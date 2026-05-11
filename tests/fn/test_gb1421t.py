"""Tests for gb1421t.gibbons_phi_cramers_v."""
import numpy as np
import pytest
from morie.fn.gb1421t import gibbons_phi_cramers_v


def test_gb1421t_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_phi_cramers_v(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb1421t_edge():
    """Test edge cases."""
    result = gibbons_phi_cramers_v(np.array([42.0]))
    assert result['n'] == 1
