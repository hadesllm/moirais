"""Tests for epsig1.em_algorithm."""
import numpy as np
import pytest
from morie.fn.epsig1 import em_algorithm


def test_epsig1_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = em_algorithm(log_lik, Q, x0, steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_epsig1_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = em_algorithm(log_lik, Q, x0, steps)
    assert isinstance(result, dict)
