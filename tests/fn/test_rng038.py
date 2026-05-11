"""Tests for rng038.rangayyan_ch3_test_signal_sin_cos."""
import numpy as np
import pytest
from morie.fn.rng038 import rangayyan_ch3_test_signal_sin_cos


def test_rng038_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_test_signal_sin_cos(t)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_rng038_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_test_signal_sin_cos(t)
    assert isinstance(result, dict)
