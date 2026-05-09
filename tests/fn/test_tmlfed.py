"""Tests for tmlfed.tmle_federated."""
import numpy as np
import pytest
from moirais.fn.tmlfed import tmle_federated


def test_tmlfed_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    site = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_federated(y, D, X, site)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlfed_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    site = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_federated(y, D, X, site)
    assert isinstance(result, dict)
