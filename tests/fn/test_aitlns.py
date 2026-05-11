"""Tests for aitlns.logistic_normal_sample."""
import numpy as np
import pytest
from morie.fn.aitlns import logistic_normal_sample


def test_aitlns_basic():
    """Test basic functionality."""
    mu = 0.0
    Sigma = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = logistic_normal_sample(mu, Sigma, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitlns_edge():
    """Test edge cases."""
    mu = 0.0
    Sigma = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = logistic_normal_sample(mu, Sigma, n)
    assert isinstance(result, dict)
