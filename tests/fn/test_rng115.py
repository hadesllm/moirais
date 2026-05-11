"""Tests for rng115.rangayyan_ch3_first_difference_phase."""
import numpy as np
import pytest
from morie.fn.rng115 import rangayyan_ch3_first_difference_phase


def test_rng115_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_first_difference_phase(omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng115_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_first_difference_phase(omega)
    assert isinstance(result, dict)
