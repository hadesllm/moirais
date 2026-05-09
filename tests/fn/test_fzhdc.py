"""Tests for fzhdc.fauzi_h_decomposition."""
import numpy as np
import pytest
from moirais.fn.fzhdc import fauzi_h_decomposition


def test_fzhdc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_h_decomposition(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_fzhdc_edge():
    """Test edge cases."""
    result = fauzi_h_decomposition(np.array([42.0]))
    assert result['n'] == 1
