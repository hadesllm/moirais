"""Tests for psis.pareto_smoothed_importance_sampling."""
import numpy as np
import pytest
from moirais.fn.psis import pareto_smoothed_importance_sampling


def test_psis_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = pareto_smoothed_importance_sampling(log_lik)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_psis_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = pareto_smoothed_importance_sampling(log_lik)
    assert isinstance(result, dict)
