"""Tests for laplc.laplace_mechanism."""
import numpy as np
import pytest
from moirais.fn.laplc import laplace_mechanism


def test_laplc_basic():
    """Test basic functionality."""
    f_value = np.random.default_rng(42).normal(0, 1, 100)
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = laplace_mechanism(f_value, sensitivity, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_laplc_edge():
    """Test edge cases."""
    f_value = np.random.default_rng(42).normal(0, 1, 100)
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = laplace_mechanism(f_value, sensitivity, epsilon)
    assert isinstance(result, dict)
