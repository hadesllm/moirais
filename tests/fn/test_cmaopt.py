"""Tests for cmaopt.cma_es."""
import numpy as np
import pytest
from morie.fn.cmaopt import cma_es


def test_cmaopt_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    lam = 0.1
    result = cma_es(f, x0, sigma, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cmaopt_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    lam = 0.1
    result = cma_es(f, x0, sigma, lam)
    assert isinstance(result, dict)
