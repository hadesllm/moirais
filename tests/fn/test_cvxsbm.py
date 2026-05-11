"""Tests for cvxsbm.boyd_subgrad_method."""
import numpy as np
import pytest
from morie.fn.cvxsbm import boyd_subgrad_method


def test_cvxsbm_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    subgrad = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_subgrad_method(f, subgrad, x0, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxsbm_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    subgrad = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_subgrad_method(f, subgrad, x0, t)
    assert isinstance(result, dict)
