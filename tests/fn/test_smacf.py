"""Tests for smacf.smacof_algorithm."""
import numpy as np
import pytest
from morie.fn.smacf import smacof_algorithm


def test_smacf_basic():
    """Test basic functionality."""
    delta = np.random.default_rng(42).normal(0, 1, 100)
    n_dims = 2
    weights = np.random.default_rng(45).exponential(1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = smacof_algorithm(delta, n_dims, weights, max_iter, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smacf_edge():
    """Test edge cases."""
    delta = np.random.default_rng(42).normal(0, 1, 100)
    n_dims = 2
    weights = np.random.default_rng(45).exponential(1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = smacof_algorithm(delta, n_dims, weights, max_iter, eps)
    assert isinstance(result, dict)
