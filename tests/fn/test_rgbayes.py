"""Tests for rgbayes.rangayyan_bayes_classifier."""
import numpy as np
import pytest
from moirais.fn.rgbayes import rangayyan_bayes_classifier


def test_rgbayes_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    class_priors = np.random.default_rng(42).normal(0, 1, 100)
    class_means = np.random.default_rng(42).normal(0, 1, 100)
    class_covs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bayes_classifier(X, class_priors, class_means, class_covs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgbayes_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    class_priors = np.random.default_rng(42).normal(0, 1, 100)
    class_means = np.random.default_rng(42).normal(0, 1, 100)
    class_covs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bayes_classifier(X, class_priors, class_means, class_covs)
    assert isinstance(result, dict)
