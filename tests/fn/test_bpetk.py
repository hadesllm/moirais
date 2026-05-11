"""Tests for bpetk.bpe_tokenizer."""
import numpy as np
import pytest
from morie.fn.bpetk import bpe_tokenizer


def test_bpetk_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab_size = 100
    result = bpe_tokenizer(corpus, vocab_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bpetk_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab_size = 100
    result = bpe_tokenizer(corpus, vocab_size)
    assert isinstance(result, dict)
