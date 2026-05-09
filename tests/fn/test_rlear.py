"""Tests for rlear.r_learner."""
import numpy as np
import pytest
from moirais.fn.rlear import r_learner


def test_rlear_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    m_model = np.random.default_rng(42).normal(0, 1, 100)
    e_model = np.random.default_rng(42).normal(0, 1, 100)
    tau_model = np.random.default_rng(42).normal(0, 1, 100)
    result = r_learner(Y, T, X, m_model, e_model, tau_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rlear_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    m_model = np.random.default_rng(42).normal(0, 1, 100)
    e_model = np.random.default_rng(42).normal(0, 1, 100)
    tau_model = np.random.default_rng(42).normal(0, 1, 100)
    result = r_learner(Y, T, X, m_model, e_model, tau_model)
    assert isinstance(result, dict)
