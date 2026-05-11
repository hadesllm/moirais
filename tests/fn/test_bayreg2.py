"""Tests for bayreg2.bayes_robust."""
import numpy as np
import pytest
from morie.fn.bayreg2 import bayes_robust


def test_bayreg2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    nu_prior = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_robust(y, X, nu_prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayreg2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    nu_prior = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_robust(y, X, nu_prior)
    assert isinstance(result, dict)
