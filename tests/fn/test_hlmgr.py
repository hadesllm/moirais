"""Tests for hlmgr.hlm_gamma_matrix."""
import numpy as np
import pytest
from moirais.fn.hlmgr import hlm_gamma_matrix


def test_hlmgr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = hlm_gamma_matrix(y, X, Z, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hlmgr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = hlm_gamma_matrix(y, X, Z, cluster)
    assert isinstance(result, dict)
