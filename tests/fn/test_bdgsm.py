"""Tests for bdgsm.bridge_sampling_marginal."""
import numpy as np
import pytest
from morie.fn.bdgsm import bridge_sampling_marginal


def test_bdgsm_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = bridge_sampling_marginal(log_lik, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bdgsm_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = bridge_sampling_marginal(log_lik, prior)
    assert isinstance(result, dict)
