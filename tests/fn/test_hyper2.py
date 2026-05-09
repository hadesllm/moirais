"""Tests for hyper2.hyperparam_optim_gp."""
import numpy as np
import pytest
from moirais.fn.hyper2 import hyperparam_optim_gp


def test_hyper2_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = hyperparam_optim_gp(X, y, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hyper2_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = hyperparam_optim_gp(X, y, prior)
    assert isinstance(result, dict)
