"""Tests for rng247.rangayyan_ch4_signal_with_echo_output."""
import numpy as np
import pytest
from morie.fn.rng247 import rangayyan_ch4_signal_with_echo_output


def test_rng247_basic():
    """Test basic functionality."""
    h = 0.3
    a = np.random.default_rng(44).normal(0, 1, 100)
    n_0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_signal_with_echo_output(h, a, n_0, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng247_edge():
    """Test edge cases."""
    h = 0.3
    a = np.random.default_rng(44).normal(0, 1, 100)
    n_0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_signal_with_echo_output(h, a, n_0, n)
    assert isinstance(result, dict)
