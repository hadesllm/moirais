"""Tests for otbarfree.ot_barycenter_free."""
import numpy as np
import pytest
from moirais.fn.otbarfree import ot_barycenter_free


def test_otbarfree_basic():
    """Test basic functionality."""
    X_list = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    n_supp = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_barycenter_free(X_list, weights, n_supp, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otbarfree_edge():
    """Test edge cases."""
    X_list = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    n_supp = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_barycenter_free(X_list, weights, n_supp, max_iter)
    assert isinstance(result, dict)
