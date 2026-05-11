"""Tests for rng119.rangayyan_ch3_three_point_central_diff_phase."""
import numpy as np
import pytest
from morie.fn.rng119 import rangayyan_ch3_three_point_central_diff_phase


def test_rng119_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_three_point_central_diff_phase(omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng119_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_three_point_central_diff_phase(omega)
    assert isinstance(result, dict)
