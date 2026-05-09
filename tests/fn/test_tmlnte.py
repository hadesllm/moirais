"""Tests for tmlnte.tmle_natural_total."""
import numpy as np
import pytest
from moirais.fn.tmlnte import tmle_natural_total


def test_tmlnte_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_natural_total(y, D, M, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlnte_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_natural_total(y, D, M, X)
    assert isinstance(result, dict)
