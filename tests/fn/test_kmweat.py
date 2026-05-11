"""Tests for kmweat.kamath_weat_bias_score."""
import numpy as np
import pytest
from morie.fn.kmweat import kamath_weat_bias_score


def test_kmweat_basic():
    """Test basic functionality."""
    X_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    Y_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    A_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    B_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_weat_bias_score(X_embeddings, Y_embeddings, A_embeddings, B_embeddings)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_kmweat_edge():
    """Test edge cases."""
    X_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    Y_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    A_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    B_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_weat_bias_score(X_embeddings, Y_embeddings, A_embeddings, B_embeddings)
    assert isinstance(result, dict)
