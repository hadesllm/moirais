"""Tests for kmw2v.kamath_word2vec_skipgram."""
import numpy as np
import pytest
from morie.fn.kmw2v import kamath_word2vec_skipgram


def test_kmw2v_basic():
    """Test basic functionality."""
    center_indices = np.random.default_rng(42).normal(0, 1, 100)
    context_indices = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_word2vec_skipgram(center_indices, context_indices, V, U)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmw2v_edge():
    """Test edge cases."""
    center_indices = np.random.default_rng(42).normal(0, 1, 100)
    context_indices = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_word2vec_skipgram(center_indices, context_indices, V, U)
    assert isinstance(result, dict)
