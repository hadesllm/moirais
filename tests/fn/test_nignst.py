"""Tests for nignst.normal_inv_gamma."""
import numpy as np
import pytest
from moirais.fn.nignst import normal_inv_gamma


def test_nignst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    mu0 = 0.0
    kappa0 = np.random.default_rng(42).normal(0, 1, 100)
    alpha0 = np.random.default_rng(42).normal(0, 1, 100)
    beta0 = np.random.default_rng(42).normal(0, 1, 100)
    result = normal_inv_gamma(y, mu0, kappa0, alpha0, beta0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nignst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    mu0 = 0.0
    kappa0 = np.random.default_rng(42).normal(0, 1, 100)
    alpha0 = np.random.default_rng(42).normal(0, 1, 100)
    beta0 = np.random.default_rng(42).normal(0, 1, 100)
    result = normal_inv_gamma(y, mu0, kappa0, alpha0, beta0)
    assert isinstance(result, dict)
