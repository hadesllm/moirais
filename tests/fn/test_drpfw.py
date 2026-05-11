"""Tests for drpfw.dropout_forward."""
import numpy as np
import pytest
from morie.fn.drpfw import dropout_forward


def test_drpfw_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = dropout_forward(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_drpfw_edge():
    """Test edge cases."""
    result = dropout_forward(np.array([42.0]))
    assert result['n'] == 1
