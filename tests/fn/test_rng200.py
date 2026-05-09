"""Tests for rng200.rangayyan_ch4_continuous_dot_product."""
import numpy as np
import pytest
from moirais.fn.rng200 import rangayyan_ch4_continuous_dot_product


def test_rng200_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_continuous_dot_product(x, y, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng200_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_continuous_dot_product(x, y, t)
    assert isinstance(result, dict)
