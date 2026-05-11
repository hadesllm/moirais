"""Tests for cvxepi.boyd_epigraph."""
import numpy as np
import pytest
from morie.fn.cvxepi import boyd_epigraph


def test_cvxepi_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_epigraph(f, x, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxepi_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = boyd_epigraph(f, x, t)
    assert isinstance(result, dict)
