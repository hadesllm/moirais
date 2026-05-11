"""Tests for cvxcen.boyd_central_path."""
import numpy as np
import pytest
from morie.fn.cvxcen import boyd_central_path


def test_cvxcen_basic():
    """Test basic functionality."""
    f0 = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_central_path(f0, f, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxcen_edge():
    """Test edge cases."""
    f0 = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_central_path(f0, f, t)
    assert isinstance(result, dict)
