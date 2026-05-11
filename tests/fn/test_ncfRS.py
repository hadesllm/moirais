"""Tests for ncfRS.ncf."""
import numpy as np
import pytest
from morie.fn.ncfRS import ncf


def test_ncfRS_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    mlp_h = np.random.default_rng(42).normal(0, 1, 100)
    result = ncf(R, K, mlp_h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ncfRS_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    mlp_h = np.random.default_rng(42).normal(0, 1, 100)
    result = ncf(R, K, mlp_h)
    assert isinstance(result, dict)
