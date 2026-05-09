"""Tests for rgdfa.rangayyan_dfa."""
import numpy as np
import pytest
from moirais.fn.rgdfa import rangayyan_dfa


def test_rgdfa_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_dfa(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgdfa_edge():
    """Test edge cases."""
    result = rangayyan_dfa(np.array([42.0]))
    assert result['n'] == 1
