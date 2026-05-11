"""Tests for causdmliv.causal_dml_iv."""
import numpy as np
import pytest
from morie.fn.causdmliv import causal_dml_iv


def test_causdmliv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_folds = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_dml_iv(y, D, Z, X, n_folds)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causdmliv_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_folds = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_dml_iv(y, D, Z, X, n_folds)
    assert isinstance(result, dict)
