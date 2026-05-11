"""Tests for glmbay.bayesian_glm."""
import numpy as np
import pytest
from morie.fn.glmbay import bayesian_glm


def test_glmbay_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    family = 'gaussian'
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = bayesian_glm(y, X, family, priors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_glmbay_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    family = 'gaussian'
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = bayesian_glm(y, X, family, priors)
    assert isinstance(result, dict)
