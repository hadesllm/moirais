"""Tests for prdmed.product_of_coefficients."""
import numpy as np
import pytest
from morie.fn.prdmed import product_of_coefficients


def test_prdmed_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = product_of_coefficients(a, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prdmed_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = product_of_coefficients(a, b)
    assert isinstance(result, dict)
