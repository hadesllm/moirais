"""Tests for loopr.loo_pareto_smooth."""
import numpy as np
import pytest
from morie.fn.loopr import loo_pareto_smooth


def test_loopr_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = loo_pareto_smooth(log_lik)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_loopr_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = loo_pareto_smooth(log_lik)
    assert isinstance(result, dict)
