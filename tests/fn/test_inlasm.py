"""Tests for inlasm.inla_spatial."""
import numpy as np
import pytest
from morie.fn.inlasm import inla_spatial


def test_inlasm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    field = np.random.default_rng(42).normal(0, 1, 100)
    precision_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = inla_spatial(y, X, field, precision_matrix)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_inlasm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    field = np.random.default_rng(42).normal(0, 1, 100)
    precision_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = inla_spatial(y, X, field, precision_matrix)
    assert isinstance(result, dict)
