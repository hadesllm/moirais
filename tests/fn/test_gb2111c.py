"""Tests for gb2111c.gibbons_elementary_coverage_beta."""
import numpy as np
import pytest
from morie.fn.gb2111c import gibbons_elementary_coverage_beta


def test_gb2111c_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_elementary_coverage_beta(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb2111c_edge():
    """Test edge cases."""
    result = gibbons_elementary_coverage_beta(np.array([42.0]))
    assert result['n'] == 1
