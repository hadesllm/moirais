"""Tests for rgeqn2.rangayyan_ch2_multivariate."""
import numpy as np
import pytest
from moirais.fn.rgeqn2 import rangayyan_ch2_multivariate


def test_rgeqn2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_ch2_multivariate(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgeqn2_edge():
    """Test edge cases."""
    result = rangayyan_ch2_multivariate(np.array([42.0]))
    assert result['n'] == 1
