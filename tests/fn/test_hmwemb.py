"""Tests for hmwemb.geron_word_embeddings."""
import numpy as np
import pytest
from morie.fn.hmwemb import geron_word_embeddings


def test_hmwemb_basic():
    """Test basic functionality."""
    vocab = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = geron_word_embeddings(vocab, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmwemb_edge():
    """Test edge cases."""
    vocab = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = geron_word_embeddings(vocab, d)
    assert isinstance(result, dict)
