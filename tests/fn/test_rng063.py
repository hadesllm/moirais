"""Tests for rng063.rangayyan_ch3_complex_exponential."""
import numpy as np
import pytest
from morie.fn.rng063 import rangayyan_ch3_complex_exponential


def test_rng063_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_complex_exponential(omega, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng063_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_complex_exponential(omega, t)
    assert isinstance(result, dict)
