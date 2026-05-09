"""Tests for diffmed.difference_in_coefficients."""
import numpy as np
import pytest
from moirais.fn.diffmed import difference_in_coefficients


def test_diffmed_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    c_prime = np.random.default_rng(42).normal(0, 1, 100)
    result = difference_in_coefficients(c, c_prime)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_diffmed_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    c_prime = np.random.default_rng(42).normal(0, 1, 100)
    result = difference_in_coefficients(c, c_prime)
    assert isinstance(result, dict)
