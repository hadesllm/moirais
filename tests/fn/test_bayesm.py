"""Tests for bayesm.dp_bayesian_mechanism."""
import numpy as np
import pytest
from morie.fn.bayesm import dp_bayesian_mechanism


def test_bayesm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    posterior_sample = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_bayesian_mechanism(y, posterior_sample, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayesm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    posterior_sample = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_bayesian_mechanism(y, posterior_sample, epsilon)
    assert isinstance(result, dict)
