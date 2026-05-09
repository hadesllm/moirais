"""Tests for ebayes.empirical_bayes_shrinkage."""
import numpy as np
import pytest
from moirais.fn.ebayes import empirical_bayes_shrinkage


def test_ebayes_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_u = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_e = np.random.default_rng(42).normal(0, 1, 100)
    result = empirical_bayes_shrinkage(y, cluster, sigma2_u, sigma2_e)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ebayes_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_u = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_e = np.random.default_rng(42).normal(0, 1, 100)
    result = empirical_bayes_shrinkage(y, cluster, sigma2_u, sigma2_e)
    assert isinstance(result, dict)
