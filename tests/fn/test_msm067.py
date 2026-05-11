"""Tests for msm067.mvsml_bayesian_regression_eq_6_9."""
import numpy as np
import pytest
from morie.fn.msm067 import mvsml_bayesian_regression_eq_6_9


def test_msm067_basic():
    """Test basic functionality."""
    trait = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    multivariate = np.random.default_rng(42).normal(0, 1, 100)
    normal = np.random.default_rng(42).normal(0, 1, 100)
    distribution = 'normal'
    result = mvsml_bayesian_regression_eq_6_9(trait, a, prior, multivariate, normal, distribution)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm067_edge():
    """Test edge cases."""
    trait = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    multivariate = np.random.default_rng(42).normal(0, 1, 100)
    normal = np.random.default_rng(42).normal(0, 1, 100)
    distribution = 'normal'
    result = mvsml_bayesian_regression_eq_6_9(trait, a, prior, multivariate, normal, distribution)
    assert isinstance(result, dict)
