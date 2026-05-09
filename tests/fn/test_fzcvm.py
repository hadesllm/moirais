"""Tests for fzcvm.fauzi_cvm_smoothed."""
import numpy as np
import pytest
from moirais.fn.fzcvm import fauzi_cvm_smoothed


def test_fzcvm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_cvm_smoothed(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_fzcvm_edge():
    """Test edge cases."""
    result = fauzi_cvm_smoothed(np.array([42.0]))
    assert result['n'] == 1
