"""Tests for facov.factor_analytic_covariance."""
import numpy as np
import pytest
from moirais.fn.facov import factor_analytic_covariance


def test_facov_basic():
    """Test basic functionality."""
    n_env = np.random.default_rng(42).normal(0, 1, 100)
    n_factors = np.random.default_rng(42).normal(0, 1, 100)
    result = factor_analytic_covariance(n_env, n_factors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_facov_edge():
    """Test edge cases."""
    n_env = np.random.default_rng(42).normal(0, 1, 100)
    n_factors = np.random.default_rng(42).normal(0, 1, 100)
    result = factor_analytic_covariance(n_env, n_factors)
    assert isinstance(result, dict)
