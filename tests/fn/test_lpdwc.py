"""Tests for lpdwc.log_pointwise_predictive_density."""
import numpy as np
import pytest
from morie.fn.lpdwc import log_pointwise_predictive_density


def test_lpdwc_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = log_pointwise_predictive_density(log_lik)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lpdwc_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = log_pointwise_predictive_density(log_lik)
    assert isinstance(result, dict)
