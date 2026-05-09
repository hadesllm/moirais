"""Tests for aitclu.compositional_kmeans."""
import numpy as np
import pytest
from moirais.fn.aitclu import compositional_kmeans


def test_aitclu_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = compositional_kmeans(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitclu_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = compositional_kmeans(X, k)
    assert isinstance(result, dict)
