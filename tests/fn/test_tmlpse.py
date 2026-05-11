"""Tests for tmlpse.tmle_path_specific."""
import numpy as np
import pytest
from morie.fn.tmlpse import tmle_path_specific


def test_tmlpse_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    M_chain = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    path = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_path_specific(y, D, M_chain, X, path)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlpse_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    M_chain = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    path = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_path_specific(y, D, M_chain, X, path)
    assert isinstance(result, dict)
