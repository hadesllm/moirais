"""Tests for meta1l.meta_learner_ensemble."""
import numpy as np
import pytest
from moirais.fn.meta1l import meta_learner_ensemble


def test_meta1l_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    result = meta_learner_ensemble(y, D, X, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_meta1l_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    result = meta_learner_ensemble(y, D, X, weights)
    assert isinstance(result, dict)
