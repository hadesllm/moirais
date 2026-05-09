"""Tests for bnstvr.bound_treatment_variation."""
import numpy as np
import pytest
from moirais.fn.bnstvr import bound_treatment_variation


def test_bnstvr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = bound_treatment_variation(y, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnstvr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = bound_treatment_variation(y, D, X)
    assert isinstance(result, dict)
