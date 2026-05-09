"""Tests for tmlnie.tmle_natural_indirect."""
import numpy as np
import pytest
from moirais.fn.tmlnie import tmle_natural_indirect


def test_tmlnie_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_natural_indirect(y, D, M, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlnie_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_natural_indirect(y, D, M, X)
    assert isinstance(result, dict)
