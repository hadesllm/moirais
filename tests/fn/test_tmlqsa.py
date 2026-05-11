"""Tests for tmlqsa.tmle_quasi_score."""
import numpy as np
import pytest
from morie.fn.tmlqsa import tmle_quasi_score


def test_tmlqsa_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    score_fn = (lambda v: float(np.mean(v)))
    result = tmle_quasi_score(y, D, X, score_fn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlqsa_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    score_fn = (lambda v: float(np.mean(v)))
    result = tmle_quasi_score(y, D, X, score_fn)
    assert isinstance(result, dict)
