"""Tests for swigl.swiglu_activation."""
import numpy as np
import pytest
from moirais.fn.swigl import swiglu_activation


def test_swigl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = swiglu_activation(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_swigl_edge():
    """Test edge cases."""
    result = swiglu_activation(np.array([42.0]))
    assert result['n'] == 1
