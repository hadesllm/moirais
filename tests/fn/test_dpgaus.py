"""Tests for dpgaus.dp_gaussian_mechanism."""
import numpy as np
import pytest
from moirais.fn.dpgaus import dp_gaussian_mechanism


def test_dpgaus_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_gaussian_mechanism(y, sensitivity, epsilon, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpgaus_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_gaussian_mechanism(y, sensitivity, epsilon, delta)
    assert isinstance(result, dict)
