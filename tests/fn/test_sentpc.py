"""Tests for sentpc.sentencepiece."""
import numpy as np
import pytest
from morie.fn.sentpc import sentencepiece


def test_sentpc_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab_size = 100
    alpha = 0.05
    result = sentencepiece(corpus, vocab_size, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sentpc_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab_size = 100
    alpha = 0.05
    result = sentencepiece(corpus, vocab_size, alpha)
    assert isinstance(result, dict)
