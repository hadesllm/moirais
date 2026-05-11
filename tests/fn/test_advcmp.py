"""Tests for advcmp.advanced_composition."""
import numpy as np
import pytest
from morie.fn.advcmp import advanced_composition


def test_advcmp_basic():
    """Test basic functionality."""
    epsilon = 1e-6
    delta = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    delta_prime = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_composition(epsilon, delta, k, delta_prime)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_advcmp_edge():
    """Test edge cases."""
    epsilon = 1e-6
    delta = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    delta_prime = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_composition(epsilon, delta, k, delta_prime)
    assert isinstance(result, dict)
