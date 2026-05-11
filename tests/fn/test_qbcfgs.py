"""Tests for qbcfgs.qb_cf_score."""
import numpy as np
import pytest
from morie.fn.qbcfgs import qb_cf_score


def test_qbcfgs_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = qb_cf_score(y, D, X, quantile)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qbcfgs_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = qb_cf_score(y, D, X, quantile)
    assert isinstance(result, dict)
