"""Tests for gb_are3.gibbons_are_dbl_exp."""
import numpy as np
import pytest
from morie.fn.gb_are3 import gibbons_are_dbl_exp


def test_gb_are3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_are_dbl_exp(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb_are3_edge():
    """Test edge cases."""
    result = gibbons_are_dbl_exp(np.array([42.0]))
    assert result['n'] == 1
