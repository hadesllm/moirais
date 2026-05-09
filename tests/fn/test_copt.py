"""Tests for copt.t_copula."""
import numpy as np
import pytest
from moirais.fn.copt import t_copula


def test_copt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    rho = 0.5
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = t_copula(y, u, v, rho, nu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_copt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    rho = 0.5
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = t_copula(y, u, v, rho, nu)
    assert isinstance(result, dict)
