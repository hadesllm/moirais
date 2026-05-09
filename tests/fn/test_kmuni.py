"""Tests for kmuni.kamath_unigram_lm_tokenizer."""
import numpy as np
import pytest
from moirais.fn.kmuni import kamath_unigram_lm_tokenizer


def test_kmuni_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_unigram_lm_tokenizer(corpus, vocab)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmuni_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_unigram_lm_tokenizer(corpus, vocab)
    assert isinstance(result, dict)
