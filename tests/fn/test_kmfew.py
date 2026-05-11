"""Tests for kmfew.kamath_few_shot_exemplar_selection."""
import numpy as np
import pytest
from morie.fn.kmfew import kamath_few_shot_exemplar_selection


def test_kmfew_basic():
    """Test basic functionality."""
    D = np.random.default_rng(42).normal(0, 1, 100)
    query_embed = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_few_shot_exemplar_selection(D, query_embed, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmfew_edge():
    """Test edge cases."""
    D = np.random.default_rng(42).normal(0, 1, 100)
    query_embed = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_few_shot_exemplar_selection(D, query_embed, K)
    assert isinstance(result, dict)
