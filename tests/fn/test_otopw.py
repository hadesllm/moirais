"""Tests for otopw.ot_optimised_potentials_warm."""
import numpy as np
import pytest
from moirais.fn.otopw import ot_optimised_potentials_warm


def test_otopw_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    f0 = np.random.default_rng(42).normal(0, 1, 100)
    g0 = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_optimised_potentials_warm(a, b, C, epsilon, f0, g0, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otopw_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    f0 = np.random.default_rng(42).normal(0, 1, 100)
    g0 = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_optimised_potentials_warm(a, b, C, epsilon, f0, g0, max_iter)
    assert isinstance(result, dict)
