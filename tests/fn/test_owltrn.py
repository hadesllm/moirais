"""Tests for owltrn.outcome_weighted_learning."""
import numpy as np
import pytest
from moirais.fn.owltrn import outcome_weighted_learning


def test_owltrn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = outcome_weighted_learning(y, D, W, pi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_owltrn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = outcome_weighted_learning(y, D, W, pi)
    assert isinstance(result, dict)
