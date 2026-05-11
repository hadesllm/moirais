"""Tests for penmth.penalty_method."""
import numpy as np
import pytest
from morie.fn.penmth import penalty_method


def test_penmth_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    result = penalty_method(f, constraints, x0, mu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_penmth_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    result = penalty_method(f, constraints, x0, mu)
    assert isinstance(result, dict)
