"""Tests for bysid.bayesian_ideal_points."""
import numpy as np
import pytest
from morie.fn.bysid import bayesian_ideal_points


def test_bysid_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bayesian_ideal_points(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_bysid_edge():
    """Test edge cases."""
    result = bayesian_ideal_points(np.array([42.0]))
    assert result['n'] == 1
