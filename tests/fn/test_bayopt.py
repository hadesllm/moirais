"""Tests for bayopt.bayesian_optimization."""
import numpy as np
import pytest
from moirais.fn.bayopt import bayesian_optimization


def test_bayopt_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    domain = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    acquisition = np.random.default_rng(42).normal(0, 1, 100)
    result = bayesian_optimization(f, domain, kernel, acquisition)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayopt_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    domain = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    acquisition = np.random.default_rng(42).normal(0, 1, 100)
    result = bayesian_optimization(f, domain, kernel, acquisition)
    assert isinstance(result, dict)
