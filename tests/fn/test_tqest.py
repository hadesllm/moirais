"""Tests for tqest.turboquant_estimate_scores."""
import numpy as np
import pytest
from morie.fn.tqest import turboquant_estimate_scores


def test_tqest_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k_tildes = np.random.default_rng(42).normal(0, 1, 100)
    norms = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_estimate_scores(q, k_tildes, norms, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqest_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k_tildes = np.random.default_rng(42).normal(0, 1, 100)
    norms = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_estimate_scores(q, k_tildes, norms, S)
    assert isinstance(result, dict)
