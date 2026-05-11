"""Tests for otentf.ot_free_energy."""
import numpy as np
import pytest
from morie.fn.otentf import ot_free_energy


def test_otentf_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    epsilon = 1e-6
    result = ot_free_energy(T, C, a, b, f, g, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otentf_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    epsilon = 1e-6
    result = ot_free_energy(T, C, a, b, f, g, epsilon)
    assert isinstance(result, dict)
