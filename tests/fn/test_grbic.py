"""Tests for grbic.geron_bic_gmm."""
import numpy as np
import pytest
from morie.fn.grbic import geron_bic_gmm


def test_grbic_basic():
    """Test basic functionality."""
    log_likelihood = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    n_params = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bic_gmm(log_likelihood, n, n_params)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grbic_edge():
    """Test edge cases."""
    log_likelihood = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    n_params = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bic_gmm(log_likelihood, n, n_params)
    assert isinstance(result, dict)
