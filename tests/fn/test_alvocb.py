"""Tests for alvocb.alammar_tokenizer_vocab_overlap."""
import numpy as np
import pytest
from morie.fn.alvocb import alammar_tokenizer_vocab_overlap


def test_alvocb_basic():
    """Test basic functionality."""
    vocab_a = np.random.default_rng(42).normal(0, 1, 100)
    vocab_b = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_tokenizer_vocab_overlap(vocab_a, vocab_b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alvocb_edge():
    """Test edge cases."""
    vocab_a = np.random.default_rng(42).normal(0, 1, 100)
    vocab_b = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_tokenizer_vocab_overlap(vocab_a, vocab_b)
    assert isinstance(result, dict)
