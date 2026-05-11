"""Tests for abcnnt.abc_neural."""
import numpy as np
import pytest
from morie.fn.abcnnt import abc_neural


def test_abcnnt_basic():
    """Test basic functionality."""
    sim = np.random.default_rng(42).normal(0, 1, 100)
    obs = np.random.default_rng(42).normal(0, 1, 100)
    theta_prior = np.random.default_rng(42).normal(0, 1, 100)
    n_train = np.random.default_rng(42).normal(0, 1, 100)
    result = abc_neural(sim, obs, theta_prior, n_train)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_abcnnt_edge():
    """Test edge cases."""
    sim = np.random.default_rng(42).normal(0, 1, 100)
    obs = np.random.default_rng(42).normal(0, 1, 100)
    theta_prior = np.random.default_rng(42).normal(0, 1, 100)
    n_train = np.random.default_rng(42).normal(0, 1, 100)
    result = abc_neural(sim, obs, theta_prior, n_train)
    assert isinstance(result, dict)
