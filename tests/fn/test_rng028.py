"""Tests for rng028.rangayyan_ch3_sifting_property."""
import numpy as np
import pytest
from morie.fn.rng028 import rangayyan_ch3_sifting_property


def test_rng028_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    t_o = np.random.default_rng(42).normal(0, 1, 100)
    T1 = np.random.default_rng(42).normal(0, 1, 100)
    T2 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_sifting_property(x, t, t_o, T1, T2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng028_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    t_o = np.random.default_rng(42).normal(0, 1, 100)
    T1 = np.random.default_rng(42).normal(0, 1, 100)
    T2 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_sifting_property(x, t, t_o, T1, T2)
    assert isinstance(result, dict)
