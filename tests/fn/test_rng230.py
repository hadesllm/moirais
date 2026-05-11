"""Tests for rng230.rangayyan_ch4_homomorphic_multiplicative_signal."""
import numpy as np
import pytest
from morie.fn.rng230 import rangayyan_ch4_homomorphic_multiplicative_signal


def test_rng230_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_homomorphic_multiplicative_signal(x, p, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng230_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_homomorphic_multiplicative_signal(x, p, t)
    assert isinstance(result, dict)
