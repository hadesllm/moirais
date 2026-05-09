"""Tests for tmlric.tmle_rare_outcome."""
import numpy as np
import pytest
from moirais.fn.tmlric import tmle_rare_outcome


def test_tmlric_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    prevalence = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_rare_outcome(y, D, X, prevalence)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlric_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    prevalence = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_rare_outcome(y, D, X, prevalence)
    assert isinstance(result, dict)
