"""Tests for poispr.poisson_predictive."""
import numpy as np
import pytest
from moirais.fn.poispr import poisson_predictive


def test_poispr_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = poisson_predictive(counts, alpha, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_poispr_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = poisson_predictive(counts, alpha, beta)
    assert isinstance(result, dict)
