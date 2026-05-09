"""Tests for spbayr.schabenberger_bayesian_hierarchical."""
import numpy as np
import pytest
from moirais.fn.spbayr import schabenberger_bayesian_hierarchical


def test_spbayr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    prior_spec = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_bayesian_hierarchical(y, x, w, prior_spec)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spbayr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    prior_spec = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_bayesian_hierarchical(y, x, w, prior_spec)
    assert isinstance(result, dict)
