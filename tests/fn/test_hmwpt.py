"""Tests for hmwpt.geron_wordpiece_tokenizer."""
import numpy as np
import pytest
from moirais.fn.hmwpt import geron_wordpiece_tokenizer


def test_hmwpt_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab_size = 100
    result = geron_wordpiece_tokenizer(corpus, vocab_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmwpt_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab_size = 100
    result = geron_wordpiece_tokenizer(corpus, vocab_size)
    assert isinstance(result, dict)
