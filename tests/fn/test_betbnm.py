"""Tests for betbnm.beta_binomial."""
import numpy as np
import pytest
from moirais.fn.betbnm import beta_binomial


def test_betbnm_basic():
    """Test basic functionality."""
    successes = np.random.default_rng(42).normal(0, 1, 100)
    trials = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = beta_binomial(successes, trials, alpha, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_betbnm_edge():
    """Test edge cases."""
    successes = np.random.default_rng(42).normal(0, 1, 100)
    trials = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = beta_binomial(successes, trials, alpha, beta)
    assert isinstance(result, dict)
