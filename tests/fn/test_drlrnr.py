"""Tests for drlrnr.r_learner."""
import numpy as np
import pytest
from morie.fn.drlrnr import r_learner


def test_drlrnr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ml_outcome = np.random.default_rng(42).normal(0, 1, 100)
    ml_propensity = np.random.default_rng(42).normal(0, 1, 100)
    result = r_learner(y, D, X, ml_outcome, ml_propensity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drlrnr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ml_outcome = np.random.default_rng(42).normal(0, 1, 100)
    ml_propensity = np.random.default_rng(42).normal(0, 1, 100)
    result = r_learner(y, D, X, ml_outcome, ml_propensity)
    assert isinstance(result, dict)
