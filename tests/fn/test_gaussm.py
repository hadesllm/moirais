"""Tests for gaussm.gaussian_mechanism."""
import numpy as np
import pytest
from moirais.fn.gaussm import gaussian_mechanism


def test_gaussm_basic():
    """Test basic functionality."""
    f_value = np.random.default_rng(42).normal(0, 1, 100)
    l2_sens = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_mechanism(f_value, l2_sens, epsilon, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gaussm_edge():
    """Test edge cases."""
    f_value = np.random.default_rng(42).normal(0, 1, 100)
    l2_sens = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_mechanism(f_value, l2_sens, epsilon, delta)
    assert isinstance(result, dict)
