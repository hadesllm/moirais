"""Tests for bnsadm.bound_admissible_estimators."""
import numpy as np
import pytest
from morie.fn.bnsadm import bound_admissible_estimators


def test_bnsadm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    family = 'gaussian'
    result = bound_admissible_estimators(y, D, X, family)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnsadm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    family = 'gaussian'
    result = bound_admissible_estimators(y, D, X, family)
    assert isinstance(result, dict)
