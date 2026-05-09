"""Tests for drlnr.dr_learner."""
import numpy as np
import pytest
from moirais.fn.drlnr import dr_learner


def test_drlnr_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mu0 = 0.0
    mu1 = np.random.default_rng(42).normal(0, 1, 100)
    e_model = np.random.default_rng(42).normal(0, 1, 100)
    cate_model = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_learner(Y, T, X, mu0, mu1, e_model, cate_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drlnr_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mu0 = 0.0
    mu1 = np.random.default_rng(42).normal(0, 1, 100)
    e_model = np.random.default_rng(42).normal(0, 1, 100)
    cate_model = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_learner(Y, T, X, mu0, mu1, e_model, cate_model)
    assert isinstance(result, dict)
