"""Tests for kmfst.kamath_fasttext_subword."""
import numpy as np
import pytest
from morie.fn.kmfst import kamath_fasttext_subword


def test_kmfst_basic():
    """Test basic functionality."""
    word = np.random.default_rng(42).normal(0, 1, 100)
    ngram_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    n_min = 0
    n_max = 100
    result = kamath_fasttext_subword(word, ngram_embeddings, n_min, n_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmfst_edge():
    """Test edge cases."""
    word = np.random.default_rng(42).normal(0, 1, 100)
    ngram_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    n_min = 0
    n_max = 100
    result = kamath_fasttext_subword(word, ngram_embeddings, n_min, n_max)
    assert isinstance(result, dict)
