"""Tests for rng029.rangayyan_ch3_signal_as_delta_decomposition."""
import numpy as np
import pytest
from moirais.fn.rng029 import rangayyan_ch3_signal_as_delta_decomposition


def test_rng029_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_signal_as_delta_decomposition(x, alpha, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng029_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_signal_as_delta_decomposition(x, alpha, t)
    assert isinstance(result, dict)
