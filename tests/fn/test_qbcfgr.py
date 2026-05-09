"""Tests for qbcfgr.quantile_balanced_cf."""
import numpy as np
import pytest
from moirais.fn.qbcfgr import quantile_balanced_cf


def test_qbcfgr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = quantile_balanced_cf(y, D, X, quantile)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qbcfgr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = quantile_balanced_cf(y, D, X, quantile)
    assert isinstance(result, dict)
