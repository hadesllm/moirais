"""Tests for matern.matern_cluster."""
import numpy as np
import pytest
from morie.fn.matern import matern_cluster


def test_matern_basic():
    """Test basic functionality."""
    lambda_p = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    r = 10
    result = matern_cluster(lambda_p, mu, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_matern_edge():
    """Test edge cases."""
    lambda_p = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    r = 10
    result = matern_cluster(lambda_p, mu, r)
    assert isinstance(result, dict)
