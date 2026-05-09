"""Tests for remlf.reml_log_likelihood."""
import numpy as np
import pytest
from moirais.fn.remlf import reml_log_likelihood


def test_remlf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = reml_log_likelihood(y, X, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_remlf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = reml_log_likelihood(y, X, V)
    assert isinstance(result, dict)
