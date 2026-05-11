"""Tests for eslism.esl_isomap."""
import numpy as np
import pytest
from morie.fn.eslism import esl_isomap


def test_eslism_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    neighbors = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_isomap(X, k, neighbors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslism_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    neighbors = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_isomap(X, k, neighbors)
    assert isinstance(result, dict)
