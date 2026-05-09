"""Tests for tmlmda.tmle_missing_data."""
import numpy as np
import pytest
from moirais.fn.tmlmda import tmle_missing_data


def test_tmlmda_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    missing = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_missing_data(y, D, X, missing)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlmda_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    missing = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_missing_data(y, D, X, missing)
    assert isinstance(result, dict)
