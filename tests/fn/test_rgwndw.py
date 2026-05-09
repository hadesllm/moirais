"""Tests for rgwndw.rangayyan_window_functions."""
import numpy as np
import pytest
from moirais.fn.rgwndw import rangayyan_window_functions


def test_rgwndw_basic():
    """Test basic functionality."""
    N = 100
    window_type = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_window_functions(N, window_type)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgwndw_edge():
    """Test edge cases."""
    N = 100
    window_type = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_window_functions(N, window_type)
    assert isinstance(result, dict)
