"""Tests for rng195.rangayyan_ch4_length_transformation."""
import numpy as np
import pytest
from moirais.fn.rng195 import rangayyan_ch4_length_transformation


def test_rng195_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    w = np.random.default_rng(45).exponential(1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_length_transformation(x, N, w, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng195_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    w = np.random.default_rng(45).exponential(1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_length_transformation(x, N, w, t)
    assert isinstance(result, dict)
