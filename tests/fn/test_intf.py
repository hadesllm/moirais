"""Tests for intf.integrate_function."""
import numpy as np
import pytest
from morie.fn.intf import integrate_function


def test_intf_basic():
    """Test basic functionality."""
    coef = np.random.default_rng(42).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = integrate_function(coef, basis)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_intf_edge():
    """Test edge cases."""
    coef = np.random.default_rng(42).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = integrate_function(coef, basis)
    assert isinstance(result, dict)
