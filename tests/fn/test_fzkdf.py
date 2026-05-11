"""Tests for fzkdf.fauzi_kdfe_properties."""
import numpy as np
import pytest
from morie.fn.fzkdf import fauzi_kdfe_properties


def test_fzkdf_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_kdfe_properties(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_fzkdf_edge():
    """Test edge cases."""
    result = fauzi_kdfe_properties(np.array([42.0]))
    assert result['n'] == 1
