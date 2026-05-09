"""Tests for causdml2.causal_dml_partial_lin."""
import numpy as np
import pytest
from moirais.fn.causdml2 import causal_dml_partial_lin


def test_causdml2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_folds = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_dml_partial_lin(y, D, X, n_folds)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causdml2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_folds = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_dml_partial_lin(y, D, X, n_folds)
    assert isinstance(result, dict)
