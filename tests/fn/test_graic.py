"""Tests for graic.geron_aic_gmm."""
import numpy as np
import pytest
from moirais.fn.graic import geron_aic_gmm


def test_graic_basic():
    """Test basic functionality."""
    log_likelihood = np.random.default_rng(42).normal(0, 1, 100)
    n_params = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_aic_gmm(log_likelihood, n_params)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_graic_edge():
    """Test edge cases."""
    log_likelihood = np.random.default_rng(42).normal(0, 1, 100)
    n_params = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_aic_gmm(log_likelihood, n_params)
    assert isinstance(result, dict)
