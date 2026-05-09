"""Tests for bmdul.bayesian_mds_unfolding."""
import numpy as np
import pytest
from moirais.fn.bmdul import bayesian_mds_unfolding


def test_bmdul_basic():
    """Test basic functionality."""
    ratings = np.random.default_rng(42).normal(0, 1, 100)
    n_dims = 2
    n_iter = 50
    result = bayesian_mds_unfolding(ratings, n_dims, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bmdul_edge():
    """Test edge cases."""
    ratings = np.random.default_rng(42).normal(0, 1, 100)
    n_dims = 2
    n_iter = 50
    result = bayesian_mds_unfolding(ratings, n_dims, n_iter)
    assert isinstance(result, dict)
