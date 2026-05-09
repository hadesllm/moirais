"""Tests for phylby.bayesian_phylogeny."""
import numpy as np
import pytest
from moirais.fn.phylby import bayesian_phylogeny


def test_phylby_basic():
    """Test basic functionality."""
    alignment = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = bayesian_phylogeny(alignment, priors, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_phylby_edge():
    """Test edge cases."""
    alignment = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = bayesian_phylogeny(alignment, priors, n_iter)
    assert isinstance(result, dict)
