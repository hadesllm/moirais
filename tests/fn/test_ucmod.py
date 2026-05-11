"""Tests for ucmod.unobserved_components."""
import numpy as np
import pytest
from morie.fn.ucmod import unobserved_components


def test_ucmod_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = unobserved_components(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ucmod_edge():
    """Test edge cases."""
    result = unobserved_components(np.array([42.0]))
    assert result['n'] == 1
