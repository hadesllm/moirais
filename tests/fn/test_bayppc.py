"""Tests for bayppc.posterior_predictive_check."""
import numpy as np
import pytest
from morie.fn.bayppc import posterior_predictive_check


def test_bayppc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_rep = np.random.default_rng(42).normal(0, 1, 100)
    statistic = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive_check(y, y_rep, statistic)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayppc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_rep = np.random.default_rng(42).normal(0, 1, 100)
    statistic = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive_check(y, y_rep, statistic)
    assert isinstance(result, dict)
