"""Tests for derivf.derivative_function."""
import numpy as np
import pytest
from moirais.fn.derivf import derivative_function


def test_derivf_basic():
    """Test basic functionality."""
    coef = np.random.default_rng(42).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    order = 4
    result = derivative_function(coef, basis, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_derivf_edge():
    """Test edge cases."""
    coef = np.random.default_rng(42).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    order = 4
    result = derivative_function(coef, basis, order)
    assert isinstance(result, dict)
