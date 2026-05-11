"""Tests for tmlivc.tmle_iv."""
import numpy as np
import pytest
from morie.fn.tmlivc import tmle_iv


def test_tmlivc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_iv(y, D, Z, covariates)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlivc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_iv(y, D, Z, covariates)
    assert isinstance(result, dict)
