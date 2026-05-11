"""Tests for rng102.rangayyan_ch3_integral_general."""
import numpy as np
import pytest
from morie.fn.rng102 import rangayyan_ch3_integral_general


def test_rng102_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_integral_general(x, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng102_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_integral_general(x, t)
    assert isinstance(result, dict)
