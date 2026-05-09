"""Tests for bicg.bayesian_information_criterion."""
import numpy as np
import pytest
from moirais.fn.bicg import bayesian_information_criterion


def test_bicg_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    n_params = np.random.default_rng(42).normal(0, 1, 100)
    n_obs = np.random.default_rng(42).normal(0, 1, 100)
    result = bayesian_information_criterion(log_lik, n_params, n_obs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bicg_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    n_params = np.random.default_rng(42).normal(0, 1, 100)
    n_obs = np.random.default_rng(42).normal(0, 1, 100)
    result = bayesian_information_criterion(log_lik, n_params, n_obs)
    assert isinstance(result, dict)
