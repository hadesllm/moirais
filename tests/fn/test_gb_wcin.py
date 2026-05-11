"""Tests for gb_wcin.gibbons_concordance_incomplete."""
import numpy as np
import pytest
from morie.fn.gb_wcin import gibbons_concordance_incomplete


def test_gb_wcin_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_concordance_incomplete(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb_wcin_edge():
    """Test edge cases."""
    result = gibbons_concordance_incomplete(np.array([42.0]))
    assert result['n'] == 1
