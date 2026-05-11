"""Tests for rgff.rangayyan_form_factor."""
import numpy as np
import pytest
from morie.fn.rgff import rangayyan_form_factor


def test_rgff_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_form_factor(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgff_edge():
    """Test edge cases."""
    result = rangayyan_form_factor(np.array([42.0]))
    assert result['n'] == 1
