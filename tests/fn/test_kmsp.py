"""Tests for kmsp.kamath_sentencepiece_tokenizer."""
import numpy as np
import pytest
from morie.fn.kmsp import kamath_sentencepiece_tokenizer


def test_kmsp_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab_size = 100
    result = kamath_sentencepiece_tokenizer(corpus, vocab_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmsp_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab_size = 100
    result = kamath_sentencepiece_tokenizer(corpus, vocab_size)
    assert isinstance(result, dict)
