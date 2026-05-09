"""Tests for baymed.bayes_mediation."""
import numpy as np
import pytest
from moirais.fn.baymed import bayes_mediation


def test_baymed_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_mediation(X, M, Y, priors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_baymed_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_mediation(X, M, Y, priors)
    assert isinstance(result, dict)
