"""Tests for drblr.doubly_robust_learner."""
import numpy as np
import pytest
from morie.fn.drblr import doubly_robust_learner


def test_drblr_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mu0_model = np.random.default_rng(42).normal(0, 1, 100)
    mu1_model = np.random.default_rng(42).normal(0, 1, 100)
    e_model = np.random.default_rng(42).normal(0, 1, 100)
    result = doubly_robust_learner(Y, T, X, mu0_model, mu1_model, e_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drblr_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mu0_model = np.random.default_rng(42).normal(0, 1, 100)
    mu1_model = np.random.default_rng(42).normal(0, 1, 100)
    e_model = np.random.default_rng(42).normal(0, 1, 100)
    result = doubly_robust_learner(Y, T, X, mu0_model, mu1_model, e_model)
    assert isinstance(result, dict)
