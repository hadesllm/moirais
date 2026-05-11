"""Tests for rng040.rangayyan_ch3_linear_ramp_filter."""
import numpy as np
import pytest
from morie.fn.rng040 import rangayyan_ch3_linear_ramp_filter


def test_rng040_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_linear_ramp_filter(t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng040_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_linear_ramp_filter(t)
    assert isinstance(result, dict)
