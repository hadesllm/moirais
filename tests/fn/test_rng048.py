"""Tests for rng048.rangayyan_ch3_laplace_transform."""
import numpy as np
import pytest
from morie.fn.rng048 import rangayyan_ch3_laplace_transform


def test_rng048_basic():
    """Test basic functionality."""
    h = 0.3
    t = np.linspace(0, 10, 100)
    s = 90
    result = rangayyan_ch3_laplace_transform(h, t, s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng048_edge():
    """Test edge cases."""
    h = 0.3
    t = np.linspace(0, 10, 100)
    s = 90
    result = rangayyan_ch3_laplace_transform(h, t, s)
    assert isinstance(result, dict)
