"""Tests for hmopt.geron_optics."""
import numpy as np
import pytest
from morie.fn.hmopt import geron_optics


def test_hmopt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    min_samples = np.random.default_rng(42).normal(0, 1, 100)
    max_eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_optics(X, min_samples, max_eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmopt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    min_samples = np.random.default_rng(42).normal(0, 1, 100)
    max_eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_optics(X, min_samples, max_eps)
    assert isinstance(result, dict)
