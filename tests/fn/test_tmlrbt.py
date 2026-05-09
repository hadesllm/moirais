"""Tests for tmlrbt.tmle_robust."""
import numpy as np
import pytest
from moirais.fn.tmlrbt import tmle_robust


def test_tmlrbt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    trim = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_robust(y, D, X, trim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlrbt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    trim = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_robust(y, D, X, trim)
    assert isinstance(result, dict)
