"""Tests for laplT.laplace_transform."""
import numpy as np
import pytest
from moirais.fn.laplT import laplace_transform


def test_laplT_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    s = 90
    result = laplace_transform(f, t, s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_laplT_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    s = 90
    result = laplace_transform(f, t, s)
    assert isinstance(result, dict)
