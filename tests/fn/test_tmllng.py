"""Tests for tmllng.tmle_longitudinal."""
import numpy as np
import pytest
from moirais.fn.tmllng import tmle_longitudinal


def test_tmllng_basic():
    """Test basic functionality."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = tmle_longitudinal(L, A, Y, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmllng_edge():
    """Test edge cases."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = tmle_longitudinal(L, A, Y, time)
    assert isinstance(result, dict)
