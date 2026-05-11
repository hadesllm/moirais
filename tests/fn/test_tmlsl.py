"""Tests for tmlsl.tmle_super_learner."""
import numpy as np
import pytest
from morie.fn.tmlsl import tmle_super_learner


def test_tmlsl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    library = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_super_learner(y, D, X, library)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlsl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    library = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_super_learner(y, D, X, library)
    assert isinstance(result, dict)
