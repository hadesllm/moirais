"""Tests for rgeqn10b.rangayyan_ch10_roc_optimal."""
import numpy as np
import pytest
from moirais.fn.rgeqn10b import rangayyan_ch10_roc_optimal


def test_rgeqn10b_basic():
    """Test basic functionality."""
    fpr = np.random.default_rng(42).normal(0, 1, 100)
    tpr = np.random.default_rng(42).normal(0, 1, 100)
    cost_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch10_roc_optimal(fpr, tpr, cost_matrix, priors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeqn10b_edge():
    """Test edge cases."""
    fpr = np.random.default_rng(42).normal(0, 1, 100)
    tpr = np.random.default_rng(42).normal(0, 1, 100)
    cost_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch10_roc_optimal(fpr, tpr, cost_matrix, priors)
    assert isinstance(result, dict)
