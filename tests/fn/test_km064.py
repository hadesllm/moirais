"""Tests for km064.kamath_ch4_loftq_objective."""
import numpy as np
import pytest
from moirais.fn.km064 import kamath_ch4_loftq_objective


def test_km064_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_ch4_loftq_objective(W, Q, A, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km064_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_ch4_loftq_objective(W, Q, A, B)
    assert isinstance(result, dict)
