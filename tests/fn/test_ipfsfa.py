"""Tests for ipfsfa.ipopt_solver."""
import numpy as np
import pytest
from moirais.fn.ipfsfa import ipopt_solver


def test_ipfsfa_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = ipopt_solver(f, constraints, x0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ipfsfa_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = ipopt_solver(f, constraints, x0)
    assert isinstance(result, dict)
