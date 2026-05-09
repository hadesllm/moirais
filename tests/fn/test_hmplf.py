"""Tests for hmplf.geron_polynomial_features."""
import numpy as np
import pytest
from moirais.fn.hmplf import geron_polynomial_features


def test_hmplf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    degree = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_polynomial_features(X, degree)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmplf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    degree = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_polynomial_features(X, degree)
    assert isinstance(result, dict)
