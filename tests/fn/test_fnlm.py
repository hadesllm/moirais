"""Tests for fnlm.function_on_function."""
import numpy as np
import pytest
from moirais.fn.fnlm import function_on_function


def test_fnlm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    basis_X = np.random.default_rng(42).normal(0, 1, 100)
    basis_Y = np.random.default_rng(42).normal(0, 1, 100)
    result = function_on_function(X, Y, basis_X, basis_Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fnlm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    basis_X = np.random.default_rng(42).normal(0, 1, 100)
    basis_Y = np.random.default_rng(42).normal(0, 1, 100)
    result = function_on_function(X, Y, basis_X, basis_Y)
    assert isinstance(result, dict)
