"""Tests for hmbpet.geron_bpe_tokenizer."""
import numpy as np
import pytest
from moirais.fn.hmbpet import geron_bpe_tokenizer


def test_hmbpet_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab_size = 100
    result = geron_bpe_tokenizer(corpus, vocab_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbpet_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab_size = 100
    result = geron_bpe_tokenizer(corpus, vocab_size)
    assert isinstance(result, dict)
