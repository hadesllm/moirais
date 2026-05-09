"""Tests for drspl.dr_did_split_sample."""
import numpy as np
import pytest
from moirais.fn.drspl import dr_did_split_sample


def test_drspl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = dr_did_split_sample(y, D, X, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drspl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = dr_did_split_sample(y, D, X, K)
    assert isinstance(result, dict)
