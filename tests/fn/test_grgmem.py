"""Tests for grgmem.geron_gmm_em_step."""
import numpy as np
import pytest
from moirais.fn.grgmem import geron_gmm_em_step


def test_grgmem_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    pi = np.random.default_rng(42).normal(0, 1, 100)
    means = np.random.default_rng(42).normal(0, 1, 100)
    covars = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gmm_em_step(X, pi, means, covars)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grgmem_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    pi = np.random.default_rng(42).normal(0, 1, 100)
    means = np.random.default_rng(42).normal(0, 1, 100)
    covars = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gmm_em_step(X, pi, means, covars)
    assert isinstance(result, dict)
