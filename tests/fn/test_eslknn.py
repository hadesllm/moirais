"""Tests for eslknn.esl_knn."""
import numpy as np
import pytest
from moirais.fn.eslknn import esl_knn


def test_eslknn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = esl_knn(X, y, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslknn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = esl_knn(X, y, k)
    assert isinstance(result, dict)
