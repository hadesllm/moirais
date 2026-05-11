"""Tests for kmicl.kamath_in_context_learning_prob."""
import numpy as np
import pytest
from morie.fn.kmicl import kamath_in_context_learning_prob


def test_kmicl_basic():
    """Test basic functionality."""
    demonstrations = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_in_context_learning_prob(demonstrations, query, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmicl_edge():
    """Test edge cases."""
    demonstrations = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_in_context_learning_prob(demonstrations, query, model)
    assert isinstance(result, dict)
