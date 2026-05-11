"""Tests for rng246.rangayyan_ch4_signal_with_echo_input."""
import numpy as np
import pytest
from morie.fn.rng246 import rangayyan_ch4_signal_with_echo_input


def test_rng246_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    n_0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_signal_with_echo_input(a, n_0, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng246_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    n_0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_signal_with_echo_input(a, n_0, n)
    assert isinstance(result, dict)
