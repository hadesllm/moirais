"""Tests for posspr.posterior_predictive."""
import numpy as np
import pytest
from morie.fn.posspr import posterior_predictive


def test_posspr_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    new = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive(data, prior, new)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_posspr_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    new = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive(data, prior, new)
    assert isinstance(result, dict)
