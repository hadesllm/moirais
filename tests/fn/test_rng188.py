"""Tests for rng188.rangayyan_ch4_pan_tompkins_derivative_operator."""
import numpy as np
import pytest
from moirais.fn.rng188 import rangayyan_ch4_pan_tompkins_derivative_operator


def test_rng188_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_pan_tompkins_derivative_operator(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng188_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_pan_tompkins_derivative_operator(x, n)
    assert isinstance(result, dict)
